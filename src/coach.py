import logging
from collections import deque
from random import shuffle
import torch.distributed as dist
import torch

from tqdm import tqdm
import csv
import copy

from enums import PlayerColor
from mcts import MCTSBrain
from HiveNNet import NNetWrapper
from utils import dotdict
from typing import Deque, List, Tuple
from arena import Arena
from gameWrapper import *
import pickle
import glob

import multiprocessing as mp
import os
import numpy as np
from numpy.typing import NDArray


log = logging.getLogger(__name__)

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]

def _self_play_worker_process(args: dotdict,
    board_size: tuple[int,int],
    action_size: int,
    checkpoint_folder: str,
    checkpoint_file: str,
    seed: int,
    local_rank: int) -> list[TrainingExample]:
    """
    Re-run self-play in a fresh process:
    - turn off distributed
    - rebuild the network wrapper and load weights
    - call executeEpisode()
    """
    try:
        # 1. seed RNG
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 2. make a local copy of args, disable distributed
        local_args: dotdict = copy.deepcopy(args)
        local_args.distributed = False
        
        # 3. Handle CUDA device assignment for worker processes
        if local_args.cuda and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # 4. build a fresh network wrapper & load weights
        nnet = NNetWrapper(board_size, action_size, local_args)
        
        # 5. Wait a bit and retry loading if file doesn't exist yet
        import time
        max_retries = 10
        for attempt in range(max_retries):
            try:
                nnet.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.5)  # Wait 500ms before retry

        # 6. make a fresh Coach and run one episode
        coach = Coach(GameWrapper(), nnet, local_args)
        return coach.executeEpisode()
        
    except Exception as e:
        log.error(f"Worker process failed with seed {seed}: {e}")
        return []  # Return empty list on failure

class Coach:
    """
    Orchestrates self-play and learning.
    """
    def __init__(self, game: GameWrapper, nnet_wrapper: NNetWrapper, args: dotdict):
        self.game = game
        self.nnet = nnet_wrapper
        self.pnet = self.nnet  
        self.args = args

        self.mcts = MCTSBrain(self.game, self.nnet, self.args)
        self.trainExamplesHistory: List[deque[TrainingExample]] = []  
        self.numEps = args.numEps  
        self.maxlenOfQueue = args.maxlenOfQueue  
        self.tempThreshold = args.tempThreshold
        self.local_rank = dist.get_rank() if args.distributed else 0


    def random_agent_action(self, board: Board, player: int) -> int:
        """
        Ignores canonicalisation completely—
        just pick uniformly from the real board’s valid moves.
        """
        valid = self.game.getValidMoves(board)
        valid_indices = np.nonzero(valid)[0]
        return int(np.random.choice(valid_indices))

    def _mcts_core(self, board: Board, player: int, nnet: NNetWrapper) -> int:
        mcts = MCTSBrain(self.game, nnet, self.args)
        pi = mcts.getActionProb(board, temp=0)
        valid = self.game.getValidMoves(board)
        pi = pi * valid
        if pi.sum() > 0:
            return int(np.argmax(pi))
        else:
            return int(np.random.choice(np.nonzero(valid)[0]))
        
    def mcts_prev_agent_action(self, board: Board, player: int) -> int:
        """
        Wraps the *previous* network (self.pnet) so it looks like a
        Callable[[Board,int],int].
        """
        return self._mcts_core(board, player, self.pnet)

    def mcts_agent_action(self, board: Board, player: int) -> int:
        """
        Wraps the *current* network (self.nnet) so it looks like a
        Callable[[Board,int],int].
        """
        return self._mcts_core(board, player, self.nnet)
    
    def executeEpisodePool(self, n_games_per_rank: int = 1) -> List[TrainingExample]:
        """Run N games in parallel processes and collect examples."""
        seeds = [self.local_rank * 10000 + i for i in range(n_games_per_rank)]
        board_size = self.nnet.board_size  if not hasattr(self.nnet, 'module') else self.nnet.module.board_size
        action_size = self.nnet.action_size if not hasattr(self.nnet, 'module') else self.nnet.module.action_size
        ckpt_folder = self.args.checkpoint
        ckpt_file = 'temp.pth.tar'

        # Prepare args for starmap
        tasks = [
            (self.args, board_size, action_size,
             ckpt_folder, ckpt_file,
             seed, self.local_rank)
            for seed in seeds
        ]

        # Spawn pool, run in isolated processes
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_games_per_rank) as pool:
            results = pool.starmap(_self_play_worker_process, tasks)

        # flatten
        return [ex for sub in results for ex in sub]

    def executeEpisode(self) -> List[TrainingExample]:
        trainExamples: List[TrainingExample] = []
        board = self.game.getInitBoard()
        currentPlayer = 1  # 1 for White, 0 for Black.
        episodeStep = 0
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0

        while True:
            episodeStep += 1
            print("[EXE EPISODE] Original board:", board)

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
            valid_moves = self.game.getValidMoves(board)
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

        if self.args.distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            # Create a Gloo-based group for Python-object collectives
            ranks = list(range(world_size))
            self.gloo_group = dist.new_group(ranks=ranks, backend="gloo")
        else:
            world_size = 1
            rank = 0

        history_path = f"{self.args.results}/selfplay_iter_*.pkl"
        files = sorted(glob.glob(history_path))
        if files:
            with open(files[-1], "rb") as f:
                self.trainExamplesHistory = pickle.load(f)
            print(f"[R{rank}] Loaded {sum(len(d) for d in self.trainExamplesHistory)} examples from disk")

        # Evaluate initial performance against a random baseline.
        if rank == 0:
            arena = Arena(self.random_agent_action,
                self.mcts_agent_action,
                self.game)
            wins_random, wins_zero, draws = arena.playGames(5)
            logging.info('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)

        for i in range(1, numIters + 1):
            logging.info(f'[R{rank}] Starting Iteration {i} ...')
            iterationTrainExamples: Deque[TrainingExample] = deque([], maxlen=self.maxlenOfQueue)

            # Save current model state before self-play
            if rank == 0:
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            
            # Wait for rank 0 to save checkpoint before all ranks start self-play
            if self.args.distributed:
                dist.barrier()

            eps_per_rank = self.args.numEps // world_size
            for _ in range(eps_per_rank):
                examples = self.executeEpisodePool()
                if examples:
                    iterationTrainExamples.extend(examples)

            # Gather examples from all ranks
            if self.args.distributed:
                all_examples = [None] * world_size

                # Gather over Gloo so we don’t hit NCCL broadcast timeouts
                dist.all_gather_object(
                    all_examples,
                    list(iterationTrainExamples),
                    group=self.gloo_group
                )

                combined_iteration = []
                for sub in all_examples:
                    if sub:
                        combined_iteration.extend(sub)
            else:
                combined_iteration = list(iterationTrainExamples)

            if rank == 0:
                self.trainExamplesHistory.append(combined_iteration)
                if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                    self.trainExamplesHistory.pop(0)

                # combine history
                trainExamples = []
                for ex in self.trainExamplesHistory:
                    trainExamples.extend(ex)
                shuffle(trainExamples)
            
                converted_examples = [(b, p, float(o)) for (b, p, o) in trainExamples]
                
                # Load previous model for comparison
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                
                self.nnet.train(trainExamples)      # Train on collected examples
                
                logging.info('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(self.mcts_prev_agent_action,   # player1 ⇒ old net
                    self.mcts_agent_action,                  # player2 ⇒ new net
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
                        self.mcts_agent_action,     # new net
                        self.game)
                    wins_random, wins_zero, draws = arena.playGames(50)
                    logging.info('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
                    with open(f"{self.args.results}/random_baseline.csv", 'a') as outfile:
                        csvwriter = csv.writer(outfile)
                        csvwriter.writerow([model_iteration, wins_zero, wins_random])
                logging.info("Iteration %d completed. Collected examples: %d", i, len(converted_examples))

                with open(f"{self.args.results}/selfplay_iter_{i}.pkl", "wb") as f:
                    pickle.dump(self.trainExamplesHistory, f)

            if self.args.distributed:
                dist.barrier()

        if self.args.distributed:
            dist.destroy_process_group()
