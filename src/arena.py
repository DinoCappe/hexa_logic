import logging
from tqdm import tqdm
from typing import Optional, Tuple
from board import Board
from enums import PlayerColor
from utils import dotdict
from gameWrapper import GameWrapper

from typing import Callable
Player = Callable[[Board, int], int]

log = logging.getLogger(__name__)

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

            valid = self.game.getValidMoves(board, curPlayer)
            if valid[action] == 0:
                raise ValueError(f"Illegal action {action}")

            # advance the real game
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

    def playGames(self, num: int, verbose: bool = False) -> Tuple[int,int,int]:
        # play half the games with (player1,player2) and half with roles swapped
        n = num // 2
        oneWon = twoWon = draws = 0

        for _ in tqdm(range(n), desc="Arena first half"):
            r = self.playGame(verbose)
            if   r ==  1: oneWon += 1
            elif r == -1: twoWon += 1
            else:          draws  += 1

        # swap who “is” player1 and who “is” player2
        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(n), desc="Arena second half"):
            r = self.playGame(verbose)
            # now a “+1” means what used to be player2 won
            if   r ==  1: twoWon += 1
            elif r == -1: oneWon += 1
            else:          draws  += 1

        return oneWon, twoWon, draws
