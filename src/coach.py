import logging
from collections import deque
from random import shuffle

from tqdm import tqdm
import csv

from mcts import MCTSBrain
from HiveNNet import NNetWrapper
from utils import dotdict
from typing import Deque
from arena import Arena
from gameWrapper import *


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

    def executeEpisode(self) -> List[TrainingExample]:
        trainExamples: List[TrainingExample] = []
        board = self.game.getInitBoard()
        currentPlayer = 1  # 1 for White, 0 for Black.
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
                print("Number of training examples generated: ", len(trainExamples))
            
            action = np.random.choice(len(pi), p=pi)
            print("[EXECUTE EPISODE] Randomly selected action: ", action)
            board, currentPlayer = self.game.getNextState(board, currentPlayer, action)
            r = self.game.getGameEnded(board, currentPlayer)
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
        nmcts = MCTSBrain(self.game, self.nnet, self.args)
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
            pmcts = MCTSBrain(self.game, self.nnet, self.args)
            
            self.nnet.train(converted_examples)
            nmcts = MCTSBrain(self.game, self.nnet, self.args)
            
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

