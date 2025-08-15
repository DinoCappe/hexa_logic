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

EPS = 1e-8

class TimeoutException(Exception): pass

@contextmanager
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
    def __init__(self, game: Optional[GameWrapper] = None, nnet: Optional[NNetWrapper] = None, args: Optional[dotdict] = None):
        self.game = game or GameWrapper()
        self.nnet = nnet
        self.args = args
        self.Qsa: Dict[Tuple[str, int], float] = {}
        self.Nsa: Dict[Tuple[str, int], int] = {}
        self.Ns: Dict[str, int] = {}
        self.Ps: Dict[str, NDArray[np.float64]] = {}
        self.Es: Dict[str, float] = {}
        self.Vs: Dict[str, NDArray[np.float64]] = {}

    def getActionProb(self, rawBoard: Board, temp: float = 1, max_time: Optional[float] = None) -> NDArray[np.float64]:
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
        s = rawBoard.stringRepresentation() if player == 1 else rawBoard.invert_colors().stringRepresentation()

        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
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
        # add openings
        probs = self.getActionProb(board, temp=0, max_time=max_time)
        valid_moves = self.game.getValidMoves(board)
        masked_probs = probs * valid_moves
        if masked_probs.sum() > 0:
            best_action_index = int(np.argmax(masked_probs))
        else:
            valid_indices = np.nonzero(valid_moves)[0]
            best_action_index = int(np.random.choice(valid_indices))
        return board.decode_move_index(best_action_index)

    def search(self, rawBoard: Board) -> float:
        player = 1 if rawBoard.current_player_color == PlayerColor.WHITE else 0
        s = rawBoard.stringRepresentation() if player == 1 else rawBoard.invert_colors().stringRepresentation()

        outcome = self.game.getGameEnded(rawBoard, player)
        if outcome != 0:
            return -outcome

        # Leaf node expansion
        if s not in self.Ps:
            canon = rawBoard if player == 1 else rawBoard.invert_colors()
            p_logits, v = self.nnet.predict(canon)
            valids = self.game.getValidMoves(rawBoard)
            p = p_logits * valids
            # Normalize for synonyms
            synonyms = self.game.getSynonymSpace(rawBoard)
            for pos, value in enumerate(synonyms):
                if p[pos] > 0 and value != pos and value is not None:
                    p[value] += p[pos]
                    p[pos] = 0.0
            p_sum = p.sum()
            self.Ps[s] = p / p_sum if p_sum > 0 else valids / valids.sum()
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # Retrieve cached
        valids = self.Vs[s] = self.game.getValidMoves(rawBoard)

        # Initialize best action from first valid index
        valid_idxs = [i for i,v in enumerate(valids) if v]
        assert valid_idxs, f"No valid moves at state {s}!"
        best_a = valid_idxs[0]
        best_u = self.Qsa.get((s,best_a), 0) + self.args.cpuct * self.Ps[s][best_a] * math.sqrt(self.Ns[s] + EPS)/(1 + self.Nsa.get((s,best_a), 0))

        # Select action with highest UCB
        for a in valid_idxs:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s,a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s])/(1 + self.Nsa[(s,a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
            if u > best_u:
                best_u, best_a = u, a

        # Recur
        next_raw, _ = self.game.getNextState(rawBoard, player, best_a)
        v = self.search(next_raw)

        # Backprop
        if (s, best_a) in self.Qsa:
            self.Qsa[(s,best_a)] = (self.Nsa[(s,best_a)] * self.Qsa[(s,best_a)] + v) / (self.Nsa[(s,best_a)] + 1)
            self.Nsa[(s,best_a)] += 1
        else:
            self.Qsa[(s,best_a)] = v
            self.Nsa[(s,best_a)] = 1
        self.Ns[s] += 1

        return -v
