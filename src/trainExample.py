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
import multiprocessing as mp
from utils import dotdict
from tqdm import tqdm
from pathlib import Path

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]
_parser = None

def _init_worker(path: str, checkpoint_folder: str, checkpoint_file: str, args):
    # force CPU inference in workers
    args = dotdict({ **args, 'cuda': False, 'distributed': False })
    game = GameWrapper()
    action_size = game.getActionSize()
    # load a CPU-only net once
    nnet = NNetWrapper(board_size=(26,26), action_size=action_size, args=args)
    ckpt = os.path.join(checkpoint_folder, checkpoint_file)
    if os.path.exists(ckpt):
        try:
            nnet.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)
            logging.info(f"Worker loaded checkpoint {ckpt}")
        except Exception as e:
            logging.warning(f"Worker failed to load checkpoint {ckpt}: {e}, starting from scratch")
    else:
        logging.warning(f"Worker did not find checkpoint {ckpt}, starting from scratch")
    # instantiate the parser
    global _parser
    _parser = TrainExample(path, game, nnet)

def _worker_parse_file(filename: str) -> list[TrainingExample]:
    # this runs *in* the worker, where _parser has been set by initializer
    return _parser._parse_file(filename)

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
        full = os.path.join(self.path, filename)
        with open(full, 'r', encoding='utf-8') as f:
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
            for b, p in symmetries:
                train_examples.append((b, p, value))

            self.example_counter += 1

            # Advance game
            try:
                board, _ = self.game.getNextState(board, player, action, line)
            except Exception as e:
                print(f"[{game_file}] Failed to apply move '{line}' at ply {ply}: {e}, stopping game.")
                break

        return train_examples

    def execute_training(self):
        cache_file = os.path.join(self.path, "pretraining_examples.pt")

        if os.path.exists(cache_file):
            print(f"[execute_training] Loading cached examples from {cache_file}…")
            all_examples = torch.load(cache_file)
        else:
            print(f"[execute_training] Parsing {len(self.file_list)} games with {mp.cpu_count()} workers")
            chunksize = max(1, len(self.file_list) // (mp.cpu_count() * 4))

            all_sublists = []
            with mp.Pool(
                processes=mp.cpu_count(),
                initializer=_init_worker,
                initargs=(self.path, self.nnet.args.checkpoint, "pre_training.pth.tar", self.nnet.args)
            ) as pool:
                for sublist in tqdm(
                    pool.imap_unordered(_worker_parse_file, self.file_list, chunksize=chunksize),
                    total=len(self.file_list),
                    desc="Parsing human games"
                ):
                    if sublist:
                        all_sublists.append(sublist)

            # flatten and cache
            all_examples = [ex for sub in all_sublists for ex in sub]
            print(f"[execute_training] Parsed {len(all_examples)} examples; saving to cache…")
            torch.save(all_examples, cache_file)

        if not all_examples:
            print("[execute_training] No examples found, skipping training")
            return

        shuffle(all_examples)
        print(f"[execute_training] Training on {len(all_examples)} examples")
        self.nnet.train(all_examples)

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
        wins_random, wins_zero, draws = arena.playGames(10)
        print('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
        logging.info(f'ZERO/RANDOM WINS : {wins_zero} / {wins_random} ; DRAWS : {draws}')