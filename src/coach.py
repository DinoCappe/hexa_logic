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

from multiprocessing import get_context
from inference_server import InferenceServer
from gpu_client import GPUClient, RemoteNNet
from functools import partial
from HiveNNet import HiveNNet


log = logging.getLogger(__name__)

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]

_G_IN_Q = None
_G_ARGS = None
_G_BOARD_SIZE = None
_G_ACTION_SIZE = None
_G_GPU_CLIENT = None
_G_WORKER_IDX = None  # optional for debug

def _pool_init(q_in, reply_queues, counter, counter_lock, args, board_size, action_size):
    # Atomically reserve an index
    with counter_lock:
        idx = counter.value
        counter.value = idx + 1
    idx = idx % max(1, len(reply_queues))

    reply_q = reply_queues[idx]

    from gpu_client import GPUClient
    global _G_IN_Q, _G_ARGS, _G_BOARD_SIZE, _G_ACTION_SIZE, _G_GPU_CLIENT
    _G_IN_Q = q_in
    _G_ARGS = args
    _G_BOARD_SIZE = board_size
    _G_ACTION_SIZE = action_size
    _G_GPU_CLIENT = GPUClient(in_q=q_in, reply_q=reply_q, timeout_s=180.0)

def _self_play_worker_entry(seed: int):
    import copy, numpy as np, torch
    from gameWrapper import GameWrapper
    from gpu_client import RemoteNNet
    local_args = copy.deepcopy(_G_ARGS)
    local_args.distributed = False
    local_args.cuda = False
    np.random.seed(seed); torch.manual_seed(seed)

    remote_nnet = RemoteNNet(_G_BOARD_SIZE, _G_ACTION_SIZE, _G_GPU_CLIENT)
    coach = Coach(GameWrapper(), remote_nnet, local_args)
    return coach.executeEpisode()

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
        self._mgr = None

    def _start_inference_server(self, board_size, action_size):
        device = torch.device(f"cuda:{self.nnet.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        assert device.type == "cuda", "This server is meant for GPU inference."

        model_ctor = partial(
            HiveNNet, board_size, action_size,
            num_channels=self.args.num_channels,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
        )

        ckpt_path = os.path.join(self.args.checkpoint, "best.pth.tar")
        if not os.path.exists(ckpt_path):
            alt = os.path.join(self.args.checkpoint, "pretrain_last.pth.tar")
            ckpt_path = alt if os.path.exists(alt) else ""

        ctx = get_context("spawn")

        # IMPORTANT: keep a strong reference to Manager on self
        self._mgr = ctx.Manager()
        in_q  = self._mgr.Queue(maxsize=4096)
        out_q = self._mgr.Queue(maxsize=4096)

        server = InferenceServer(
            model_ctor=model_ctor,
            checkpoint_path=ckpt_path,
            device=device,
            in_q=in_q,
            out_q=out_q,
            max_batch=128,
            max_wait_ms=12.0,
            use_amp=True,
            log_prefix=f"[GPU-SERVER R{self.nnet.local_rank}]"
        )
        server.start()
        return server, in_q, out_q

    def _stop_inference_server(self, server, in_q):
        # 1) Tell server to stop
        try:
            in_q.put(None)
        except Exception:
            pass

        # 2) Wait for it to exit
        try:
            server.join(timeout=20)
            if server.is_alive():
                # last resort: terminate (shouldn't be needed normally)
                server.terminate()
                server.join(timeout=10)
        except Exception:
            pass

        # 3) Now it is safe to stop the Manager
        try:
            if getattr(self, "_mgr", None) is not None:
                self._mgr.shutdown()
        finally:
            self._mgr = None

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
        
    def executeEpisodePool(self, n_games_per_rank: int, in_q) -> list[TrainingExample]:
        seeds = [self.local_rank * 10000 + i for i in range(n_games_per_rank)]
        board_size  = self.nnet.board_size  if not hasattr(self.nnet, 'module') else self.nnet.module.board_size
        action_size = self.nnet.action_size if not hasattr(self.nnet, 'module') else self.nnet.module.action_size

        processes = n_games_per_rank
        ctx = mp.get_context('spawn')

        reply_queues = [self._mgr.Queue(maxsize=1024) for _ in range(processes)]
        counter      = self._mgr.Value('i', 0)      # ValueProxy
        counter_lock = self._mgr.Lock()             # LockProxy

        with ctx.Pool(
            processes=processes,
            initializer=_pool_init,
            initargs=(in_q, reply_queues, counter, counter_lock, self.args, board_size, action_size),
        ) as pool:
            results = pool.map(_self_play_worker_entry, seeds)

        return [ex for sub in results for ex in sub]

    def executeEpisode(self) -> List[TrainingExample]:
        trainExamples: List[TrainingExample] = []
        board = self.game.getInitBoard()
        currentPlayer = 1  # 1 for White, 0 for Black.
        episodeStep = 0
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0

        while True:
            episodeStep += 1
            # print("[EXE EPISODE] Original board:", board)

            temp = 1 if episodeStep < self.tempThreshold else 0

            # Get move probabilities from MCTS.
            pi = self.mcts.getActionProb(board, temp=temp)
            symmetries = self.game.getSymmetries(board, pi)
            for b, p in symmetries:
                # b is now an NDArray[np.float64] (the encoded board)
                trainExamples.append((b, p, 0.0))  # Use a placeholder 0.0 instead of None.
                # print("Number of training examples generated: ", len(trainExamples))
            
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

            # print("[EXECUTE EPISODE] Randomly selected action: ", action)
            board, currentPlayer = self.game.getNextState(board, player, action)
            r = self.game.getGameEnded(board, player)
            if r != 0:
                # Now update outcome for all examples.
                return [(b, p, r * ((-1) ** (1 != currentPlayer))) for (b, p, _) in trainExamples]
            
    def learn(self):
        """
        Iterative self-play + training loop with a per-rank GPU inference server.
        - Each rank starts one CUDA inference server (batched predict).
        - Self-play runs in CPU workers; they query the server for NN evals.
        - We sync *per self-play batch* (frequent barriers) to keep ranks aligned.
        """
        # ----- distributed setup -----
        if self.args.distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            # a Gloo group for Python-object collectives
            ranks = list(range(world_size))
            self.gloo_group = dist.new_group(ranks=ranks, backend="gloo")
        else:
            world_size = 1
            rank = 0

        # ----- resume self-play history if present (rank-agnostic read) -----
        history_path = f"{self.args.results}/selfplay_iter_*.pkl"
        files = sorted(glob.glob(history_path))
        if files:
            with open(files[-1], "rb") as f:
                self.trainExamplesHistory = pickle.load(f)
            print(f"[R{rank}] Loaded {sum(len(d) for d in self.trainExamplesHistory)} examples from disk")

        # ----- quick baseline vs random (single-process) -----
        if rank == 0:
            arena = Arena(self.random_agent_action, self.mcts_agent_action, self.game)
            wins_random, wins_zero, draws = arena.playGames(5)
            logging.info('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
            print('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)

        # derive model sizes for the inference server
        board_size  = self.nnet.board_size  if not hasattr(self.nnet, 'module') else self.nnet.module.board_size
        action_size = self.nnet.action_size if not hasattr(self.nnet, 'module') else self.nnet.module.action_size

        numIters = self.args.numIters
        model_iteration = 1

        for i in range(1, numIters + 1):
            logging.info(f'[R{rank}] Starting Iteration {i} ...')
            print(f'[R{rank}] Starting Iteration {i} ...', flush=True)

            # ===== start the per-rank GPU inference server =====
            server, in_q, out_q = self._start_inference_server(board_size, action_size)

            try:
                # ===== self-play, batched with frequent sync =====
                eps_per_rank = self.args.numEps // world_size
                batch = 10
                done = 0

                # rank-local buffer; rank0 will merge everyone’s per-batch lists
                iterationTrainExamples_local = deque([], maxlen=self.maxlenOfQueue)

                # rank0 staging buffer for merged batch-by-batch results
                merged_this_iter_rank0 = [] if rank == 0 else None

                while done < eps_per_rank:
                    n = min(batch, eps_per_rank - done)

                    # run n games in parallel on this rank; workers query the GPU server
                    examples = self.executeEpisodePool(n_games_per_rank=n, in_q=in_q)
                    if examples:
                        iterationTrainExamples_local.extend(examples)

                    done += n
                    logging.info(f"[R{rank}] self-play progress: {done}/{eps_per_rank} games")
                    print(f"[R{rank}] self-play progress: {done}/{eps_per_rank} games", flush=True)

                    # ---- frequent sync: exchange JUST this batch ----
                    if self.args.distributed:
                        local_batch = list(examples) if examples else []

                        # 1) align ranks at batch boundary
                        dist.barrier(group=self.gloo_group)

                        # 2) gather this small python list
                        all_batches = [None] * world_size
                        dist.all_gather_object(all_batches, local_batch, group=self.gloo_group)

                        # 3) rank0 merges the batch across ranks
                        if rank == 0:
                            for sub in all_batches:
                                if sub:
                                    merged_this_iter_rank0.extend(sub)

                        # 4) optional barrier to keep tempo similar
                        dist.barrier(group=self.gloo_group)

                # ===== after self-play batches, finalize examples =====
                if self.args.distributed:
                    if rank == 0:
                        combined_iteration = merged_this_iter_rank0
                    else:
                        # non-zero ranks have already contributed per-batch;
                        # nothing more to send; keep an empty list locally.
                        combined_iteration = []
                else:
                    combined_iteration = list(iterationTrainExamples_local)

                # ===== train + evaluate on rank 0 =====
                if rank == 0:
                    # maintain rolling history
                    self.trainExamplesHistory.append(combined_iteration)
                    if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                        self.trainExamplesHistory.pop(0)

                    # flatten & shuffle
                    trainExamples = []
                    for ex in self.trainExamplesHistory:
                        trainExamples.extend(ex)
                    shuffle(trainExamples)
                    converted_examples = [(b, p, float(o)) for (b, p, o) in trainExamples]

                    # snapshot current model as "previous"
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

                    # train on the collected examples (local GPU via DDP or single)
                    logging.info(f"[R0] TRAIN on {len(converted_examples)} examples")
                    self.nnet.train(converted_examples)

                    # pit new vs previous (single-process arena)
                    logging.info('PITTING AGAINST PREVIOUS VERSION')
                    arena = Arena(self.mcts_prev_agent_action,  # player1 => old net
                                self.mcts_agent_action,       # player2 => new net
                                self.game)
                    pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
                    logging.info('NEW/PREV WINS : %d / %d ; DRAWS : %d', nwins, pwins, draws)

                    # accept / reject
                    if (pwins + nwins == 0) or (float(nwins) / max(1, (pwins + nwins)) < self.args.updateThreshold):
                        logging.info('REJECTING NEW MODEL')
                        # rollback to previous
                        self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                    else:
                        logging.info('ACCEPTING NEW MODEL')
                        model_iteration += 1
                        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

                        # quick baseline vs random
                        logging.info('PITTING AGAINST RANDOM BASELINE')
                        arena = Arena(self.random_agent_action, self.mcts_agent_action, self.game)
                        wins_random, wins_zero, draws = arena.playGames(50)
                        logging.info('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
                        with open(f"{self.args.results}/random_baseline.csv", 'a') as outfile:
                            csv.writer(outfile).writerow([model_iteration, wins_zero, wins_random])

                    logging.info("Iteration %d completed. Collected examples this iter: %d",
                                i, len(converted_examples))

                    # persist history for crash-resume
                    with open(f"{self.args.results}/selfplay_iter_{i}.pkl", "wb") as f:
                        pickle.dump(self.trainExamplesHistory, f)

            finally:
                # always stop the inference server (even on errors)
                self._stop_inference_server(server, in_q)

            # keep ranks together between iterations
            if self.args.distributed:
                dist.barrier(group=self.gloo_group)

        if self.args.distributed:
            dist.destroy_process_group()
