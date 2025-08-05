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

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]
_parser = None

def _init_worker(path: str, checkpoint_folder: str, checkpoint_file: str, args):
    # force CPU inference in workers
    args = dotdict({ **args, 'cuda': False, 'distributed': False })
    game = GameWrapper()
    action_size = game.getActionSize()
    # load a CPU-only net once
    nnet = NNetWrapper(board_size=(26,26), action_size=action_size, args=args)
    nnet.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)
    # instantiate the parser
    global _parser
    _parser = TrainExample(path, game, nnet)

class TrainExample:
    def __init__(self, path: str, game: GameWrapper, nnet: NNetWrapper):
        self.game = game
        self.nnet = nnet
        self.path = path
        self.file_list = [f for f in os.listdir(path) if f.endswith('.pgn')]
        self.example_counter = 0 
        self.diagnostics_path = os.path.join(path, "pretraining_diagnostics.csv")

        if not os.path.exists(self.diagnostics_path):
            with open(self.diagnostics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "game_file", "ply", "human_move", "pi_mcts_human",
                    "entropy", "confidence", "base_alpha", "adjusted_alpha",
                    "disagreement", "value", "top_mcts_moves"  # last column is a small summary
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
            pi_mcts: np.ndarray,
            human_action_idx: int,
            base_alpha: float,
            adjusted_alpha: float,
            value: float,
            outcome: float):
        entropy = self._entropy(pi_mcts)
        action_prob = float(pi_mcts[human_action_idx])
        max_ent = np.log(self.game.getActionSize())
        confidence = 1.0 - (entropy / max_ent)
        disagreement = 1.0 - action_prob

        # Top few MCTS moves for context (move_str:prob)
        top_k = 5
        sorted_idxs = np.argsort(-pi_mcts)[:top_k]
        best_moves = []
        for idx in sorted_idxs:
            move_str = ""  # try to decode safely (fallback to index if decode fails)
            try:
                move_str = self.game.getInitBoard().decode_move_index(idx)  # this is approximate; you could instead keep board and call decode_move_index on it
            except Exception:
                move_str = str(idx)
            best_moves.append(f"{move_str}:{pi_mcts[idx]:.3f}")
        top_mcts_summary = "|".join(best_moves)

        with open(self.diagnostics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                game_file,
                ply,
                human_move,
                f"{action_prob:.6f}",
                f"{entropy:.6f}",
                f"{confidence:.6f}",
                f"{base_alpha:.4f}",
                f"{adjusted_alpha:.4f}",
                f"{disagreement:.6f}",
                f"{value:.4f}",
                top_mcts_summary,
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
        mcts = MCTSBrain(self.game, self.nnet, self.nnet.args)
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

            # Human one-hot (with smoothing if desired)
            epsilon = 0.05
            human_pi = np.full(action_size, epsilon / (action_size - 1), dtype=np.float64)
            human_pi[action] = 1.0 - epsilon

            # MCTS policy
            pi_mcts = mcts.getActionProb(board, temp=1)

            # Base alpha schedule
            base_alpha = self._get_base_alpha(self.example_counter, max_examples=100_000, min_alpha=0.5)

            # Compute confidence and adjust alpha
            ent = self._entropy(pi_mcts)
            max_ent = np.log(action_size)
            confidence = 1.0 - (ent / max_ent)
            alpha = base_alpha
            if confidence > 0.7 and pi_mcts[action] < 0.01:
                alpha = max(0.5, alpha - 0.2)
            elif confidence < 0.3:
                alpha = min(1.0, alpha + 0.1)

            # Blend target policy
            pi_target = alpha * human_pi + (1.0 - alpha) * pi_mcts
            if pi_target.sum() > 0:
                pi_target = pi_target / pi_target.sum()
            else:
                pi_target = human_pi / human_pi.sum()

            # Value from perspective of canonical player
            value = outcome if board.current_player_color == PlayerColor.WHITE else -outcome

            # Log diagnostics
            self._log_move(
                game_file=game_file,
                ply=ply,
                human_move=line,
                pi_mcts=pi_mcts,
                human_action_idx=action,
                base_alpha=base_alpha,
                adjusted_alpha=alpha,
                value=value,
                outcome=outcome,
            )

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
        # launch a pool of workers equal to the cluster's CPU count
        path = self.path
        ckpt_folder = self.nnet.args.checkpoint
        ckpt_file = "pre_training.pth.tar"

        # spin up one CPU worker per core, each loading its own CPU net
        with mp.Pool(
            processes=mp.cpu_count(),
            initializer=_init_worker,
            initargs=(path, ckpt_folder, ckpt_file, self.nnet.args)
        ) as pool:
            for examples in pool.imap_unordered(lambda fname: _parser.parse_game_file(fname),
                                                self.file_list,
                                                chunksize=2):
                if not examples:
                    continue
                shuffle(examples)
                # now train on the GPU once per batch of examples
                self.nnet.train(examples)

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