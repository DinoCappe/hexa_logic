import logging
from collections import deque
from random import shuffle

from tqdm import tqdm
import csv

from enums import PlayerColor
from mcts import MCTSBrain
from HiveNNet import NNetWrapper
from utils import dotdict
from typing import Deque
from arena import Arena
from gameWrapper import *
from ai import Random


log = logging.getLogger(__name__)

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]

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

        self.mcts = MCTSBrain(self.game, self.nnet, self.args)
        self.trainExamplesHistory: List[deque[TrainingExample]] = []  
        self.numEps = args.numEps  
        self.maxlenOfQueue = args.maxlenOfQueue  
        self.tempThreshold = args.tempThreshold

    def random_agent_action(self, board: Board) -> int:
        nrandom = Random()
        move_str = nrandom.calculate_best_move(board)
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0
        return board.encode_move_string(move_str, player)
    
    def mcts_agent_action(self, board: Board) -> int:
        nmcts = MCTSBrain(self.game, self.nnet, self.args)
        move_str = nmcts.calculate_best_move(board)
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0
        return board.encode_move_string(move_str, player)

    def executeEpisode(self) -> List[TrainingExample]:
        trainExamples: List[TrainingExample] = []
        board = self.game.getInitBoard()
        currentPlayer = 1  # 1 for White, 0 for Black.
        episodeStep = 0
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0

        while True:
            episodeStep += 1
            print("[EXE EPISODE] Original board:", board)
            # canonicalBoard = self.game.getCanonicalForm(board, currentPlayer)
            # print("[EXE EPISODE] Canonical board:", canonicalBoard)

            temp = 1 if episodeStep < self.tempThreshold else 0

            # Get move probabilities from MCTS.
            pi = self.mcts.getActionProb(board, temp=temp)
            symmetries = self.game.getSymmetries(board, pi)
            for b, p in symmetries:
                # b is now an NDArray[np.float64] (the encoded board)
                trainExamples.append((b, p, 0.0))  # Use a placeholder 0.0 instead of None.
                print("Number of training examples generated: ", len(trainExamples))
            
            # Check that the selected action is valid.
            action = np.random.choice(len(pi), p=pi)
            valid_moves = self.game.getValidMoves(board, player)
            if valid_moves[action] == 0:
                print(f"[EXE EPISODE] Action {action} is invalid. Resampling from valid moves.")
                valid_indices = np.nonzero(valid_moves)[0]
                if valid_indices.size == 0:
                    raise ValueError("No valid moves available!")
                # Pick a random valid move.
                action = np.random.choice(valid_indices)
                print("[EXE EPISODE] Resampled action:", action)

            print("[EXECUTE EPISODE] Randomly selected action: ", action)
            board, currentPlayer = self.game.getNextState(board, player, action)
            r = self.game.getGameEnded(board, player)
            if r != 0:
                # Now update outcome for all examples.
                return [(b, p, r * ((-1) ** (1 != currentPlayer))) for (b, p, _) in trainExamples]
            
    def learn(self):
        """
        Performs iterative self-play and training.
        After each iteration, it pits the new network against the previous version.
        The new network is accepted only if it wins at or above a threshold.
        """
        numIters = self.args.numIters
        model_iteration = 1

        # Evaluate initial performance against a random baseline.
        arena = Arena(self.random_agent_action,
              self.mcts_agent_action,
              self.game)
        wins_random, wins_zero, draws = arena.playGames(5)
        logging.info('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)

        for i in range(1, numIters + 1):
            logging.info('Starting Iteration %d ...', i)
            iterationTrainExamples: Deque[TrainingExample] = deque([], maxlen=self.maxlenOfQueue)
            for _ in tqdm(range(self.numEps), desc="Self-play episodes"):
                # Reset MCTS for each self-play game.
                self.mcts = MCTSBrain(self.game, self.nnet, self.args)
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
            
            self.nnet.train(converted_examples)
            
            logging.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(self.mcts_agent_action,
                          self.mcts_agent_action,
                          self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d', nwins, pwins, draws)
            if (pwins + nwins == 0) or (float(nwins) / (pwins + nwins) < self.args.updateThreshold):
                logging.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                logging.info('ACCEPTING NEW MODEL')
                model_iteration += 1
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                logging.info('PITTING AGAINST RANDOM BASELINE')
                arena = Arena(self.random_agent_action,
                    self.mcts_agent_action,
                    self.game)
                wins_random, wins_zero, draws = arena.playGames(50)
                logging.info('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
                with open(f"{self.args.results}/random_baseline.csv", 'a') as outfile:
                    csvwriter = csv.writer(outfile)
                    csvwriter.writerow([model_iteration, wins_zero, wins_random])
            logging.info("Iteration %d completed. Collected examples: %d", i, len(converted_examples))

