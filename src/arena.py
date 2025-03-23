import logging
import time
from tqdm import tqdm
import csv
from typing import Optional, Tuple
from board import Board
from utils import dotdict
from gameWrapper import GameWrapper

from typing import Callable
Player = Callable[[Board], int]

log = logging.getLogger(__name__)

class Arena:
    """
    Arena where two agents (players) are pitted against each other.
    """
    def __init__(self, player1: Player, player2: Player, game: GameWrapper, 
                 display: Optional[Callable[[Board], None]] = None,
                 time_moves: bool = False, args: Optional[dotdict] = None):
        """
        Args:
            player1: A function that takes a canonical board and returns an action index.
            player2: A function that takes a canonical board and returns an action index.
            game: An instance of GameWrapper providing the game interface.
            display: An optional function to display the board.
            time_moves: Whether to time moves.
            args: Additional arguments.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.time_moves = time_moves
        self.args = args

    def playGame(self, verbose: bool = False) -> float:
        start: float = 0.0
        if self.time_moves:
            start = time.time()

        curPlayer = 1  # Assume player1 starts.
        board = self.game.getInitBoard()
        it = 0

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            if verbose and self.display:
                print("Turn", it, "Player", curPlayer)
                self.display(board)
            action = self.player1(canonicalBoard) if curPlayer == 1 else self.player2(canonicalBoard)
            valid_moves = self.game.getValidMoves(canonicalBoard, 1)
            if valid_moves[action] == 0:
                log.error(f"Action {action} is not valid!")
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        if verbose and self.display:
            print("Game over at turn", it, "Result", self.game.getGameEnded(board, 1))
            self.display(board)

        if self.time_moves:
            end = time.time()
            total_time = end - start
            time_per_move = total_time / it
            if self.args and 'results' in self.args:
                with open(f"{self.args.results}/usability_analysis.csv", 'a') as outfile:
                    csvwriter = csv.writer(outfile)
                    csvwriter.writerow([self.args.get('numMCTSSims', 0), time_per_move])

        return curPlayer * self.game.getGameEnded(board, curPlayer)


    def playGames(self, num: int, verbose: bool = False) -> Tuple[int, int, int]:
        """
        Plays num games (with starting player swapped halfway) and returns:
            (wins for player1, wins for player2, draws)
        """
        num_half = num // 2
        oneWon = 0
        twoWon = 0
        draws = 0

        for _ in tqdm(range(num_half), desc="Arena - first half"):
            result = self.playGame(verbose=verbose)
            if result == 1:
                oneWon += 1
            elif result == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num_half), desc="Arena - second half"):
            result = self.playGame(verbose=verbose)
            if result == 1:
                twoWon += 1
            elif result == -1:
                oneWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
