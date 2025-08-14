import logging
from typing import Optional, Tuple
from board import Board
from enums import PlayerColor
from utils import dotdict
from gameWrapper import GameWrapper

import multiprocessing as mp
from multiprocessing import Pool
from typing import Tuple, Callable, Optional
from tqdm import tqdm

_ARENA_CTX = {}  # {'nnet': NNetWrapper, 'args': dotdict}

Player = Callable[[Board, int], int]

log = logging.getLogger(__name__)

def _arena_worker_init(checkpoint_folder: str,
                       checkpoint_file: str,
                       board_size: tuple[int, int],
                       action_size: int,
                       args_dict: dict):
    """Runs once per worker process. Build a CPU-only model here."""
    import os
    from HiveNNet import NNetWrapper  # adjust import if needed
    from utils import dotdict

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU in workers
    args = dotdict({**args_dict, 'cuda': False, 'distributed': False})
    nnet = NNetWrapper(board_size=board_size, action_size=action_size, args=args)

    ckpt = os.path.join(checkpoint_folder, checkpoint_file)
    try:
        nnet.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)
        logging.info(f"[arena worker] loaded checkpoint {ckpt}")
    except Exception as e:
        logging.warning(f"[arena worker] failed to load {ckpt}: {e} (starting untrained)")

    nnet.nnet.eval()
    _ARENA_CTX['nnet'] = nnet
    _ARENA_CTX['args'] = args

def _agent_random(board, player, game):
    import numpy as np
    valid = game.getValidMoves(board)
    idxs = np.nonzero(valid)[0]
    return int(np.random.choice(idxs))

def _agent_mcts(board, player, game):
    # Build MCTS on-demand per move or keep one per game if you prefer.
    from mcts import MCTSBrain
    nnet = _ARENA_CTX['nnet']
    mcts = MCTSBrain(game, nnet, nnet.args)
    pi = mcts.getActionProb(board, temp=0)
    valid = game.getValidMoves(board)
    pi = pi * valid
    if pi.sum() > 0:
        return int(pi.argmax())
    else:
        import numpy as np
        return int(np.random.choice(np.nonzero(valid)[0]))

def _play_one_game(task) -> float:
    """
    Worker: play exactly one Arena game using worker-local CPU net.
    task = (p1_token, p2_token, display, time_moves, args_dict)
    """
    p1_token, p2_token, display, time_moves, args_dict = task

    # Fresh game per worker-run
    game = GameWrapper()
    arena = Arena(None, None, game, display, time_moves, args_dict)  # placeholders

    # Map tokens to functions that close over THIS worker's `game`
    if p1_token == "mcts":
        p1 = lambda board, player: _agent_mcts(board, player, game)
    else:
        p1 = lambda board, player: _agent_random(board, player, game)

    if p2_token == "mcts":
        p2 = lambda board, player: _agent_mcts(board, player, game)
    else:
        p2 = lambda board, player: _agent_random(board, player, game)

    arena.player1 = p1
    arena.player2 = p2
    return arena.playGame(verbose=False)

class Arena:
    """
    Arena where two agents (players) are pitted against each other.
    """
    def __init__(self,
                 player1: Player,
                 player2: Player,
                 game: GameWrapper,
                 display: Optional[Callable[[Board], None]] = None,
                 time_moves: bool = False,
                 args: Optional[dotdict] = None):
        self.player1 = player1
        self.player2 = player2
        self.game    = game
        self.display = display
        self.time_moves = time_moves
        self.args = args

    def playGame(self, verbose: bool = False) -> float:
        # 1 means “player1 (White) to move”, 0 means “player2 (Black) to move”
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        # loop until somebody wins or draws
        while True:
            result = self.game.getGameEnded(board, curPlayer)
            if result != 0:
                # from the perspective of player1=White:
                # if result==1 then White just won; if -1 Black won;
                return result

            it += 1
            curPlayer = 1 if board.current_player_color == PlayerColor.WHITE else 0
            # ask the right agent for its move
            if curPlayer==1:
                action = self.player1(board, curPlayer)
            else:
                action = self.player2(board, curPlayer)

            valid = self.game.getValidMoves(board)
            if valid[action] == 0:
                raise ValueError(f"Illegal action {action}")

            # advance the real game
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

    def playGames(self,
                num: int,
                num_workers: int = None,
                checkpoint_folder: str = None,
                checkpoint_file: str = None,
                board_size: Tuple[int, int] = None,
                action_size: int = None,
                nnet_args: Optional[dotdict] = None) -> Tuple[int,int,int]:

        num_workers = num_workers or mp.cpu_count()
        half = num // 2

        # Derive agent tokens from the callables provided to Arena __init__.
        # (Adjust names if your methods differ.)
        def _to_token(fn) -> str:
            name = getattr(fn, "__name__", "")
            return "mcts" if "mcts" in name.lower() else "random"

        p1_token = _to_token(self.player1) if self.player1 is not None else "random"
        p2_token = _to_token(self.player2) if self.player2 is not None else "random"

        tasks = []
        for _ in range(half):
            tasks.append((p1_token, p2_token, self.display, self.time_moves, self.args))
        for _ in range(num - half):
            tasks.append((p2_token, p1_token, self.display, self.time_moves, self.args))

        # Pool initializer args
        if checkpoint_folder is None or checkpoint_file is None or board_size is None or action_size is None:
            raise ValueError("playGames requires checkpoint_folder, checkpoint_file, board_size, action_size when using multiprocessing.")

        init_args = (checkpoint_folder, checkpoint_file, board_size, action_size, dict(nnet_args or {}))

        oneWon = twoWon = draws = 0
        # Safer context on CUDA machines
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers,
                    initializer=_arena_worker_init,
                    initargs=init_args) as pool:
            for r in tqdm(pool.imap_unordered(_play_one_game, tasks), total=len(tasks), desc="Arena"):
                if   r ==  1: oneWon += 1
                elif r == -1: twoWon += 1
                else:          draws  += 1

        return oneWon, twoWon, draws

