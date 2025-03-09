import logging
from collections import deque
from random import shuffle
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import csv

from board import Board
from enums import GameState, PlayerColor
from mcts import MCTSBrain
from HiveNNet import NNetWrapper
from utils import dotdict
from numpy.typing import NDArray
from typing import Deque
from arena import Arena

log = logging.getLogger(__name__)

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]

USE_SYMMETRIES = True
ACTION_SPACE_SHAPE = (14, 14)
EXTRA_ACTION = 1  # if your pi vector length is 14*14+1

class GameWrapper:
    """
    A simple wrapper that adapts our Board class to the interface expected by Coach.
    """

    def getInitBoard(self) -> Board:
        return Board()
    
    def getCanonicalForm(self, board: Board, player: int) -> Board:
        """
        Returns the board from the perspective of player 1.
        If the current player is 1, return the board as is.
        Otherwise, return a new board with colors inverted.
        """
        if player == 1:
            return board
        else:
            return board.invert_colors()

    def getNextState(self, board: Board, player: int, action: int) -> Tuple[Board, int]:
        """
        Applies the move corresponding to the action index.
        Assumes that board.valid_moves is a semicolon-separated string
        and that the ordering aligns with our action space.
        """
        valid_moves = board.valid_moves.split(";")
        if action < len(valid_moves):
            board.play(valid_moves[action])
        else:
            # If action is out-of-range, play a pass move.
            board.play("pass")
        # After playing, determine the next player.
        # We map PlayerColor to an integer: 1 for White, -1 for Black.
        new_player = 1 if board.current_player_color == PlayerColor.WHITE else -1
        return board, new_player

    def getGameEnded(self, board: Board, player: int) -> float:
        """
        Returns a numerical outcome for the game:
            0 if game is not ended,
            0.5 for a draw,
            1 if the player wins,
           -1 if the player loses.
        """
        if board.state in [GameState.NOT_STARTED, GameState.IN_PROGRESS]:
            return 0.0
        elif board.state == GameState.DRAW:
            return 0.5
        elif board.state == GameState.WHITE_WINS:
            return 1.0 if player == 1 else -1.0
        elif board.state == GameState.BLACK_WINS:
            return 1.0 if player == -1 else -1.0
        return 0.0

    def getSymmetries(self, board: Board, pi: NDArray[np.float64]) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Return a list of (board, pi) pairs for all 12 symmetries of the board.
        The board is represented as a numpy array (obtained from board.encode_board),
        and pi is a policy vector that is assumed to be spatially encoded.
        
        If symmetries are disabled or the policy vector doesn't match expectations,
        simply return the original board encoding and pi.
        """
        if not USE_SYMMETRIES:
            return [(board.encode_board(grid_size=14).astype(np.float64), pi)]

        np_board = board.encode_board(grid_size=14) # shape (channels, 14, 14)
        
        board_shape = (14, 14)
        expected_length = board_shape[0] * board_shape[1] + EXTRA_ACTION
        if pi.shape[0] != expected_length:
            return [(np_board.astype(np.float64), pi)]
        
        # Separate the extra action (if any) from the spatial part.
        pi_board = np.reshape(pi[:-EXTRA_ACTION], board_shape)
        extra = pi[-EXTRA_ACTION:]
        
        sym_list: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        cur_board = np.asarray(np_board, dtype=np.float64)  # Ensure float64
        cur_pi = np.asarray(pi_board, dtype=np.float64)
        extra = np.asarray(extra, dtype=np.float64)

        # First 6 rotations (each by 60 degrees)
        for _i in range(6):
            combined = np.concatenate((cur_pi.ravel(), extra))
            sym_list.append((cur_board, combined))

            cur_board = board.rotate_board_60(cur_board).astype(np.float64)
            cur_pi = board.rotate_pi_60(cur_pi).astype(np.float64)

        # Reflect vertically.
        cur_board = board.reflect_board_vertically(cur_board).astype(np.float64)
        cur_pi = board.reflect_pi_vertically(cur_pi).astype(np.float64)

        # Next 6 rotations of the reflected board.
        for _i in range(6):
            combined = np.concatenate((cur_pi.ravel(), extra))
            sym_list.append((cur_board, combined))
            cur_board = board.rotate_board_60(cur_board).astype(np.float64)
            cur_pi = board.rotate_pi_60(cur_pi).astype(np.float64)

        return sym_list

    def getActionSize(self) -> int:
        action_space_shape = (22, 7, 11)  # Tile-relative encoding.
        action_space_size = np.prod(action_space_shape)
        return int(action_space_size + 1)  # +1 for the pass move.
    
    def getValidMoves(self, board: Board, player: int) -> NDArray[np.float64]:
        """
        Returns a binary vector of valid moves.
        """
        valid_moves = board.valid_moves.split(";")
        action_size = self.getActionSize()
        valid = np.zeros(action_size, dtype=np.float64)
        for i, move in enumerate(valid_moves):
            if move:
                valid[i] = 1.0
        return valid

class Coach:
    """
    Orchestrates self-play and learning.
    """
    def __init__(self, game: GameWrapper, nnet_wrapper: NNetWrapper, args: dotdict):
        self.game = game
        self.nnet = nnet_wrapper
        # For simplicity, we use the same network as a competitor.
        self.pnet = self.nnet  
        self.args = args

        self.mcts = MCTSBrain(self.nnet, iterations=args.mcts_iterations, exploration_constant=args.exploration_constant)
        self.trainExamplesHistory: List[deque[TrainingExample]] = []  
        self.numEps = args.numEps  
        self.maxlenOfQueue = args.maxlenOfQueue  
        self.tempThreshold = args.tempThreshold

    def executeEpisode(self) -> List[TrainingExample]:
        trainExamples: List[TrainingExample] = []
        board = self.game.getInitBoard()
        currentPlayer = 1  # 1 for White, -1 for Black.
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, currentPlayer)
            temp = 1 if episodeStep < self.tempThreshold else 0

            # Get move probabilities from MCTS.
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            symmetries = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in symmetries:
                # b is now an NDArray[np.float64] (the encoded board)
                trainExamples.append((b, p, 0.0))  # Use a placeholder 0.0 instead of None.
            
            action = np.random.choice(len(pi), p=pi)
            board, currentPlayer = self.game.getNextState(board, currentPlayer, action)
            r = self.game.getGameEnded(board, currentPlayer)
            if r != 0:
                # Now update outcome for all examples.
                return [(b, p, r * ((-1) ** (1 != currentPlayer))) for (b, p, _) in trainExamples]
            
    def learn(self):
        """
        Performs iterative self-play and training.
        After each iteration, it pitts the new network against the previous version.
        The new network is accepted only if it wins at or above a threshold.
        """
        numIters = self.args.numIters
        model_iteration = 1

        # Evaluate initial performance against a random baseline.
        nmcts = MCTSBrain(self.nnet, iterations=self.args.mcts_iterations, exploration_constant=self.args.exploration_constant)
        arena = Arena(lambda x: self.args.random_move_baseline(x),
                      lambda x: int(np.argmax(nmcts.getActionProb(x, temp=0))),
                      self.game)
        wins_random, wins_zero, draws = arena.playGames(50)
        logging.info('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
        with open(f"{self.args.results}/random_baseline.csv", 'a') as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow([model_iteration, wins_zero, wins_random])

        for i in range(1, numIters + 1):
            logging.info('Starting Iteration %d ...', i)
            iterationTrainExamples: Deque[TrainingExample] = deque([], maxlen=self.maxlenOfQueue)
            for _ in tqdm(range(self.numEps), desc="Self-play episodes"):
                # Reset MCTS for each self-play game.
                self.mcts = MCTSBrain(self.nnet, iterations=self.args.mcts_iterations, exploration_constant=self.args.exploration_constant)
                examples = self.executeEpisode()
                iterationTrainExamples.extend(examples)
            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                logging.warning("Removing oldest training examples. History length = %d", len(self.trainExamplesHistory))
                self.trainExamplesHistory.pop(0)

            # Combine training examples from history.
            trainExamples: List[TrainingExample] = []
            for ex in self.trainExamplesHistory:
                trainExamples.extend(ex)
            shuffle(trainExamples)
            
            converted_examples = [(b, p, float(o)) for (b, p, o) in trainExamples]
            
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTSBrain(self.pnet, iterations=self.args.mcts_iterations, exploration_constant=self.args.exploration_constant)
            
            self.nnet.train(converted_examples)
            nmcts = MCTSBrain(self.nnet, iterations=self.args.mcts_iterations, exploration_constant=self.args.exploration_constant)
            
            logging.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: int(np.argmax(pmcts.getActionProb(x, temp=0))),
                          lambda x: int(np.argmax(nmcts.getActionProb(x, temp=0))),
                          self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            with open(f"{self.args.results}/stage3.csv", 'a') as outfile:
                csvwriter = csv.writer(outfile)
                csvwriter.writerow([nwins, pwins])
            logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d', nwins, pwins, draws)
            if (pwins + nwins == 0) or (float(nwins) / (pwins + nwins) < self.args.updateThreshold):
                logging.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                logging.info('ACCEPTING NEW MODEL')
                model_iteration += 1
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                logging.info('PITTING AGAINST RANDOM BASELINE')
                arena = Arena(lambda x: self.args.random_move_baseline(x),
                              lambda x: int(np.argmax(nmcts.getActionProb(x, temp=0))),
                              self.game)
                wins_random, wins_zero, draws = arena.playGames(50)
                logging.info('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
                with open(f"{self.args.results}/random_baseline.csv", 'a') as outfile:
                    csvwriter = csv.writer(outfile)
                    csvwriter.writerow([model_iteration, wins_zero, wins_random])
            logging.info("Iteration %d completed. Collected examples: %d", i, len(converted_examples))

