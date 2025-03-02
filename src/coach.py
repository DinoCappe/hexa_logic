import logging
from collections import deque
from random import shuffle
from copy import deepcopy
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from board import Board
from enums import GameState, PlayerColor
from mcts import MCTSBrain
from HiveNNet import NNetWrapper
from utils import dotdict
from numpy.typing import NDArray
from typing import Optional
from typing import Deque

log = logging.getLogger(__name__)

TrainingExample = Tuple[Board, NDArray[np.float64], Optional[float]]

class GameWrapper:
    """
    A simple wrapper that adapts our Board class to the interface expected by Coach.
    """
    def getInitBoard(self) -> Board:
        return Board()
    
    def getCanonicalForm(self, board: Board, player: int) -> Board:
        # For simplicity, assume our board is already canonical.
        return board

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

    def getSymmetries(self, board: Board, pi: NDArray[np.float64]) -> List[Tuple[Board, NDArray[np.float64]]]:
        """
        Returns a list of (board, pi) pairs for symmetric transformations.
        For now, we simply return the board and pi as is.
        """
        return [(board, pi)]
    
    def getActionSize(self) -> int:
        """
        Returns the size of the action space. Adjust this number to match your game.
        """
        return 100
    
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
        """
        Executes one self-play game.
        Returns training examples in the form (board, target_policy, target_value).
        """
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
                trainExamples.append((deepcopy(b), p, None))
            
            action = np.random.choice(len(pi), p=pi)
            board, currentPlayer = self.game.getNextState(board, currentPlayer, action)
            r = self.game.getGameEnded(board, currentPlayer)
            if r != 0:
                # Update outcome for all examples.
                # Here, we assume the outcome from the perspective of the starting player.
                return [(b, p, r * ((-1) ** (1 != currentPlayer))) for (b, p, _) in trainExamples]
            
    def learn(self):
        """
        Runs iterative self-play and training.
        """
        numIters = self.args.numIters
        _model_iteration = 1

        for i in range(1, numIters + 1):
            log.info(f"Starting Iteration {i}")
            iterationTrainExamples: Deque[TrainingExample] = deque([], maxlen=self.maxlenOfQueue)
            for _eps in tqdm(range(self.numEps), desc="Self-play episodes"):
                self.mcts = MCTSBrain(self.nnet, iterations=self.args.mcts_iterations, exploration_constant=self.args.exploration_constant)
                examples = self.executeEpisode()
                iterationTrainExamples.extend(examples)
            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)
            
            # Combine training examples from history.
            trainExamples: list[TrainingExample] = []
            for ex in self.trainExamplesHistory:
                trainExamples.extend(ex)
            shuffle(trainExamples)
            
            # Train the network with collected examples.
            # Convert each training example so that the outcome is a float (using 0.0 if None)
            converted_examples = [(b, p, float(o) if o is not None else 0.0) for (b, p, o) in trainExamples]
            self.nnet.train(converted_examples)
            log.info(f"Iteration {i} completed. Collected examples: {len(converted_examples)}")

