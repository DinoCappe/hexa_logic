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

Player = Callable[[Board, int], int]

log = logging.getLogger(__name__)

def _play_one_game(args) -> float:
    """
    Worker function to play exactly one Arena game.
    """
    player1, player2, display, time_moves, args_dict = args
    game = GameWrapper()
    arena = Arena(player1, player2, game, display, time_moves, args_dict)
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

    def playGames(self, num: int, num_workers: int = None) -> Tuple[int,int,int]:
        """
        Parallel version of playGames. Splits the `num` matches
        across a Pool of workers.
        """
        num_workers = num_workers or mp.cpu_count()
        half = num // 2

        # build the task list: first half with (p1,p2), second half with (p2,p1)
        tasks = []
        for _ in range(half):
            tasks.append((self.player1, self.player2, self.display, self.time_moves, self.args))
        # swap players
        for _ in range(num - half):
            tasks.append((self.player2, self.player1, self.display, self.time_moves, self.args))

        oneWon = twoWon = draws = 0
        with Pool(processes=num_workers) as pool:
            # use tqdm to monitor progress
            for r in tqdm(pool.imap_unordered(_play_one_game, tasks), total=len(tasks), desc="Arena"):
                if   r ==  1: oneWon += 1
                elif r == -1: twoWon += 1
                else:          draws  += 1

        return oneWon, twoWon, draws
