import math
import numpy as np
from typing import Dict, Tuple, Optional
from numpy.typing import NDArray
from board import Board
from ai import Brain
from gameWrapper import GameWrapper
from HiveNNet import NNetWrapper
from utils import dotdict
from enums import PlayerColor
import signal
from contextlib import contextmanager
# from copy import deepcopy

EPS = 1e-8
class TimeoutException(Exception): pass

@contextmanager
# Works only on Unix-like systems (Linux, macOS) and single-threaded Python programs.
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

class MCTSBrain(Brain):
    """
    Dictionary-based MCTS that caches Q-values, visit counts, 
    neural network policies, and valid moves for states.
    """
    def __init__(self, game: Optional[GameWrapper] = None, nnet: Optional[NNetWrapper] = None, args: Optional[dotdict] = None):
        """
        Args:
            game: A game interface providing methods like stringRepresentation,
                  getGameEnded, getValidMoves, getActionSize, getNextState, and getCanonicalForm.
            nnet: A neural network wrapper with a predict() method.
            args: A configuration object/dotdict containing hyperparameters.
                  Must include:
                      - numMCTSSims: number of MCTS simulations per move
                      - cpuct: the exploration constant.
            Expected calls only with optional game.
        """
        if game is None:
            # Ludo: Come vogliamo usare GameWrapper?
            self.game = GameWrapper()
        else:
            self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa: Dict[Tuple[str, int], float] = {}    # Q-values for state-action pairs
        self.Nsa: Dict[Tuple[str, int], int] = {}        # Visit counts for state-action pairs
        self.Ns: Dict[str, int] = {}                     # Visit counts for states
        self.Ps: Dict[str, NDArray[np.float64]] = {}       # Initial NN policy for states

        self.Es: Dict[str, float] = {}                   # Game ended status for states
        self.Vs: Dict[str, NDArray[np.float64]] = {}       # Valid moves for states

    def getActionProb(self, rawBoard: Board, temp: float = 1, max_time: Optional[float] = None) -> NDArray[np.float64]:
        """
        rawBoard: the actual game state
        temp: temperature
        """
        if max_time is None:
            for _ in range(self.args.numMCTSSims):
                self.search(rawBoard)
        else:
            with time_limit(max_time):
                try:
                    while True:
                        self.search(rawBoard)
                except TimeoutException:
                    pass
        player = 1 if rawBoard.current_player_color == PlayerColor.WHITE else 0
        if player == 1:
            s = rawBoard.stringRepresentation()
        else:
            s = rawBoard.invert_colors().stringRepresentation()

        counts = [self.Nsa.get((s, a), 0)
                  for a in range(self.game.getActionSize())]
        if temp == 0:
            bestAs = np.argwhere(np.array(counts) == np.max(counts)).flatten()
            bestA = int(np.random.choice(bestAs))
            probs = np.zeros(len(counts), dtype=np.float64)
            probs[bestA] = 1.0
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        total = float(sum(counts))
        if total == 0:
            return np.ones(len(counts), dtype=np.float64) / len(counts)
        return np.array([x / total for x in counts], dtype=np.float64)

    def calculate_best_move(self, board: Board, max_time: Optional[float] = None) -> str:
        """
        Runs MCTS on the given board state and returns the best move as a string.
        Uses getActionProb with temperature 0 for deterministic selection.
        """
        probs = self.getActionProb(board, temp=0, max_time=max_time)
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0
        valid_moves = self.game.getValidMoves(board, player)
        masked_probs = probs * valid_moves
        
        # If sum(masked_probs) == 0, you could choose a random move among valid indices as a fallback.
        if np.sum(masked_probs) > 0:
            best_action_index = int(np.argmax(masked_probs))
        else:
            valid_indices = np.nonzero(valid_moves)[0]
            best_action_index = int(np.random.choice(valid_indices))
        
        move_str = board.decode_move_index(player, best_action_index)
        return move_str

    def search(self, rawBoard: Board) -> float:
        """
        rawBoard: the actual game state
        Returns negative value for minimax backup.
        """
        player = 1 if rawBoard.current_player_color == PlayerColor.WHITE else 0

        if player == 1:
            s = rawBoard.stringRepresentation()
        else:
            s = rawBoard.invert_colors().stringRepresentation()

        outcome = self.game.getGameEnded(rawBoard, player)
        if outcome != 0:
            return -outcome

        if s not in self.Ps:
            # Only for the NN we need the canonical board:
            canon = rawBoard if player == 1 else rawBoard.invert_colors()
            p_logits, v = self.nnet.predict(canon)

            valids = self.game.getValidMoves(rawBoard, player)
            p = p_logits * valids
            if p.sum() > 0:
                p /= p.sum()
            else:
                p = valids / valids.sum()

            self.Ps[s] = p
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        fresh_valids = self.game.getValidMoves(rawBoard, player)
        if not np.array_equal(fresh_valids, self.Vs[s]):
            # update both our cache and local var
            self.Vs[s] = fresh_valids
            valids = fresh_valids
        else:
            valids = self.Vs[s]

        best_u, best_a = -float('inf'), -1
        for a in range(self.game.getActionSize()):
            if not valids[a]:
                continue
            if (s, a) in self.Qsa:
                u = ( self.Qsa[(s,a)]
                    + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s])
                      / (1 + self.Nsa[(s,a)]) )
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
            if u > best_u:
                best_u, best_a = u, a

        next_raw, _next_player = self.game.getNextState(rawBoard, player, best_a)
        v = self.search(next_raw)

        if (s, best_a) in self.Qsa:
            self.Qsa[(s,best_a)] = ((self.Nsa[(s,best_a)] * self.Qsa[(s,best_a)] + v)
                                     / (self.Nsa[(s,best_a)] + 1))
            self.Nsa[(s,best_a)] += 1
        else:
            self.Qsa[(s,best_a)] = v
            self.Nsa[(s,best_a)] = 1
        self.Ns[s] += 1

        return -v