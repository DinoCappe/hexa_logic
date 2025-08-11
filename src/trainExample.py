from gameWrapper import *
import re
from HiveNNet import NNetWrapper
from random import shuffle
from arena import Arena
from board import Board, PlayerColor
from mcts import MCTSBrain
import os
import csv
import logging
import numpy as np
from utils import dotdict
from tqdm import tqdm
from pathlib import Path
import torch
import time
import multiprocessing as mp  # add this at the top (you use mp.current_process())
from multiprocessing import Manager, Pool, cpu_count
from typing import Tuple
from numpy.typing import NDArray

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]

_parser = None
_progress = None

def _init_worker(path: str, checkpoint_folder: str, checkpoint_file: str, args, progress):
    global _parser, _progress
    _progress = progress
    # Safer args copy so **args doesn't rely on dotdict being a Mapping
    safe_args = dotdict(dict(args))
    safe_args.cuda = False
    safe_args.distributed = False

    game = GameWrapper()
    action_size = game.getActionSize()
    nnet = NNetWrapper(board_size=(26, 26), action_size=action_size, args=safe_args)
    ckpt = os.path.join(checkpoint_folder, checkpoint_file)
    if os.path.exists(ckpt):
        try:
            nnet.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)
            logging.info(f"Worker loaded checkpoint {ckpt}")
        except Exception as e:
            logging.warning(f"Worker failed to load checkpoint {ckpt}: {e}, starting from scratch")
    else:
        logging.warning(f"Worker did not find checkpoint {ckpt}, starting from scratch")

    _parser = TrainExample(path, game, nnet)

def _worker_parse_file(filename: str) -> list[TrainingExample]:
    proc = mp.current_process()
    print(f"[{proc.name} pid={proc.pid}] parsing {filename}", flush=True)
    out = _parser._parse_file(filename)
    # Manager().Value proxy uses get()/set(), not .value / get_lock()
    _progress.set(_progress.get() + 1)
    return out

class TrainExample:
    def __init__(self, path: str, game: GameWrapper, nnet: NNetWrapper):
        self.game = game
        self.nnet = nnet
        self.path = Path(path)
        self.file_list = [str(p) for p in self.path.rglob("*.pgn")]
        self.example_counter = 0 
        self.diagnostics_path = os.path.join(path, "pretraining_diagnostics.csv")

        if not os.path.exists(self.diagnostics_path):
            with open(self.diagnostics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "game_file", "ply", "human_move", "value", "outcome"  # last column is a small summary
                ])
    
    def _entropy(self, p: np.ndarray) -> float:
        p_safe = p + 1e-12
        return -np.sum(p_safe * np.log(p_safe))

    def _get_base_alpha(self, example_idx: int, max_examples: int = 100_000, min_alpha: float = 0.5) -> float:
        frac = min(1.0, example_idx / max_examples)
        return 1.0 - frac * (1.0 - min_alpha)
    
    def _log_move(self,
                  game_file: str,
                  ply: int,
                  human_move: str,
                  value: float,
                  outcome: float):
        """Log just the ply, human move, and value/outcome."""
        with open(self.diagnostics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                game_file,
                ply,
                human_move,
                f"{value:.4f}",
                f"{outcome:.4f}",
            ])

    def _extract_outcome(self, game_string: str) -> float:
        match = re.search(r'\[Result\s+"(.*?)"\]', game_string)
        if match:
            result = match.group(1)
            if result == "WhiteWins":
                return 1.0
            elif result == "BlackWins":
                return -1.0
            elif result == "Draw" or result == "AcceptedDraw":
                return -1e-2  # small bias
            elif result == "ResignWhite":
                return -0.95 # bias for resignation
            elif result == "ResignBlack":
                return 0.95 # bias for resignation
        return 0.0
    
    def _extract_extentions(self, game_string: str) -> bool:
        return game_string.find('Base+MLP') != -1
    
    def _parse_file(self, filename: str) -> list[TrainingExample]:
        """Worker: read one PGN, parse it into examples."""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        self.current_game_file = filename
        return self.parse_game(content)
    
    def parse_game(self, game_string: str) -> list[TrainingExample]:
        board = None
        # mcts = MCTSBrain(self.game, self.nnet, self.nnet.args)
        if self._extract_extentions(game_string):
            board = self.game.getInitBoard(expansions=True)
        else:
            board = self.game.getInitBoard()
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0
        outcome = self._extract_outcome(game_string)
        game_string = game_string[game_string.find('\n\n') + 2:]  # Skip the header
        lines = [line for line in game_string.split('\n') if line.strip()]
        train_examples: list[TrainingExample] = []
        action_size = self.game.getActionSize()

        game_file = getattr(self, "current_game_file", "unknown")  # set this before calling if available
        for ply, line in enumerate(lines, start=1):
            line = line[line.find('.') + 2:]  # Skip the move number
            try:
                action = board.encode_move_string(line)
            except Exception as e:
                print(f"[{game_file}] Failed to encode human move '{line}' at ply {ply}: {e}, skipping rest of game.")
                break

            # Canonicalize board for network
            canon = board
            if canon.current_player_color == PlayerColor.BLACK:
                canon = canon.invert_colors()

            epsilon = 0.05
            pi_target = np.full(action_size, epsilon/(action_size-1), dtype=np.float64)
            pi_target[action] = 1.0 - epsilon
            # Value from perspective of canonical player
            value = outcome if board.current_player_color == PlayerColor.WHITE else -outcome

            # Log diagnostics
            self._log_move(game_file, ply, line, value, outcome)

            # Add symmetries
            symmetries = self.game.getSymmetries(canon, pi_target)
            # when appending training examples
            for b, p in symmetries:
                b = b.astype(np.uint8, copy=False)      # board planes only need 0..7
                p = p.astype(np.float32, copy=False)    # policy
                v = np.float32(value)                   # value
                train_examples.append((b, p, v))


            self.example_counter += 1

            # Advance game
            try:
                board, _ = self.game.getNextState(board, player, action, line)
            except Exception as e:
                print(f"[{game_file}] Failed to apply move '{line}' at ply {ply}: {e}, stopping game.")
                break

        return train_examples

    def execute_training(self):
        self.path = os.path.abspath(self.path)
        shards_dir = os.path.join(self.path, "shards")
        os.makedirs(shards_dir, exist_ok=True)

        total_games = len(self.file_list)
        print(f"[execute_training] Parsing {total_games} games with {cpu_count()} workers", flush=True)

        shard_size_games = 300    # save after each ~300 files returned
        shard_id = 0
        buffer = []
        parsed_last_print = 0

        with Manager() as mgr:
            progress = mgr.Value('i', 0)
            with Pool(
                processes=12,
                initializer=_init_worker,
                initargs=(self.path, self.nnet.args.checkpoint, "pre_training.pth.tar", self.nnet.args, progress)
            ) as pool:
                for examples in pool.imap_unordered(_worker_parse_file, self.file_list, chunksize=8):
                    if examples:
                        buffer.extend(examples)

                    done = progress.get()

                    # periodic status
                    if done - parsed_last_print >= 100:
                        print(f"[execute_training] parsed {done}/{total_games} games", flush=True)
                        parsed_last_print = done

                    # flush a shard every N games parsed
                    if done >= (shard_id + 1) * shard_size_games:
                        shard_path = os.path.join(shards_dir, f"uhp_shard_{shard_id:03d}.pt")
                        torch.save(buffer, shard_path)
                        print(f"[execute_training] wrote shard {shard_id} "
                            f"({len(buffer)} examples) → {shard_path}", flush=True)
                        shard_id += 1
                        buffer.clear()

        # final shard
        if buffer:
            shard_path = os.path.join(shards_dir, f"uhp_shard_{shard_id:03d}.pt")
            torch.save(buffer, shard_path)
            print(f"[execute_training] wrote final shard {shard_id} "
                f"({len(buffer)} examples) → {shard_path}", flush=True)
            buffer.clear()

        # Now train from shards without loading everything in RAM
        print(f"[execute_training] Training from shards in {shards_dir}", flush=True)
        self.nnet.train(pretrain_dir=shards_dir)

    def mcts_agent_action(self, board: Board, player: int) -> int:
        """
        Wraps the *current* network (self.nnet) so it looks like a
        Callable[[Board,int],int].
        """
        mcts = MCTSBrain(self.game, self.nnet, self.nnet.args)
        pi = mcts.getActionProb(board, temp=0)
        valid = self.game.getValidMoves(board)
        pi = pi * valid
        if pi.sum() > 0:
            return int(np.argmax(pi))
        else:
            return int(np.random.choice(np.nonzero(valid)[0]))
    
    def random_agent_action(self, board: Board, player: int) -> int:
        """
        Ignores canonicalisation completely—
        just pick uniformly from the real board’s valid moves.
        """
        valid = self.game.getValidMoves(board)
        valid_indices = np.nonzero(valid)[0]
        return int(np.random.choice(valid_indices))

    def evaluate_training(self):
        arena = Arena(self.random_agent_action,
                self.mcts_agent_action,
                self.game)
        wins_random, wins_zero, draws = arena.playGames(10, 12)
        print('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
        logging.info(f'ZERO/RANDOM WINS : {wins_zero} / {wins_random} ; DRAWS : {draws}')