import math
import numpy as np
from typing import Dict, Tuple
from numpy.typing import NDArray
from board import Board
from ai import Brain
from gameWrapper import GameWrapper
from HiveNNet import NNetWrapper
from utils import dotdict
from typing import List, Optional

EPS = 1e-8

class MCTSBrain(Brain):
    """
    Dictionary-based MCTS that caches Q-values, visit counts, 
    neural network policies, and valid moves for states.
    """
    def __init__(self, game: GameWrapper, nnet: NNetWrapper, args: dotdict):
        """
        Args:
            game: A game interface providing methods like stringRepresentation,
                  getGameEnded, getValidMoves, getActionSize, getNextState, and getCanonicalForm.
            nnet: A neural network wrapper with a predict() method.
            args: A configuration object/dotdict containing hyperparameters.
                  Must include:
                      - numMCTSSims: number of MCTS simulations per move
                      - cpuct: the exploration constant.
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa: Dict[Tuple[str, int], float] = {}    # Q-values for state-action pairs
        self.Nsa: Dict[Tuple[str, int], int] = {}        # Visit counts for state-action pairs
        self.Ns: Dict[str, int] = {}                     # Visit counts for states
        self.Ps: Dict[str, NDArray[np.float64]] = {}       # Initial NN policy for states

        self.Es: Dict[str, float] = {}                   # Game ended status for states
        self.Vs: Dict[str, NDArray[np.float64]] = {}       # Valid moves for states

    def getActionProb(self, canonicalBoard: Board, temp: float = 1) -> NDArray[np.float64]:
        for _i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, 1)

        s = canonicalBoard.stringRepresentation()
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
        print("[GET ACTION PROB] Counts:", counts)
        
        if temp == 0:
            bestAs = np.argwhere(np.array(counts) == np.max(counts)).flatten()
            bestA = int(np.random.choice(bestAs))
            probs = np.zeros(len(counts), dtype=np.float64)
            probs[bestA] = 1.0
            #print("[GET ACTION PROB] Deterministic move chosen:", bestA)
            return probs
        counts = [x ** (1.0 / temp) for x in counts]
        total = float(sum(counts))
        if total == 0:
            print("[GET ACTION PROB] Warning: Total counts sum to zero, returning uniform distribution.")
            return np.ones(len(counts), dtype=np.float64) / len(counts)
        probs = [x / total for x in counts]
        #print("[GET ACTION PROB] Probability distribution:", probs)
        return np.array(probs, dtype=np.float64)

    def calculate_best_move(self, board: Board) -> str:
        """
        Runs MCTS on the given board state and returns the best move as a string.
        Uses getActionProb with temperature 0 for deterministic selection.
        """
        probs = self.getActionProb(board, temp=0)
        valid_moves = self.game.getValidMoves(board, 1)
        masked_probs = probs * valid_moves
        
        # If sum(masked_probs) == 0, you could choose a random move among valid indices as a fallback.
        if np.sum(masked_probs) > 0:
            best_action_index = int(np.argmax(masked_probs))
        else:
            valid_indices = np.nonzero(valid_moves)[0]
            best_action_index = int(np.random.choice(valid_indices))
        
        move_str = board.decode_move_index(1, best_action_index)
        return move_str

    def search(self, canon: Board, player: int = 1, path: Optional[List[str]] = None) -> float:
        """
        canon is always a canonicalized board (to-move is 'player 1').
        Returns the negative value of the state.
        """
        if path is None:
            path = []

        s = canon.stringRepresentation()
        print(f"{'  ' * len(path)}[SEARCH depth={len(path)}]")

        new_path = path + [s]

        # --- TERMINAL CHECK on the canonical board ---
        self.Es[s] = self.game.getGameEnded(canon, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        # --- LEAF NODE: ask NN for P, v and initialize caches ---
        if s not in self.Ps:
            p_logits, v = self.nnet.predict(canon)
            valid_mask = self.game.getValidMoves(canon, 1)

            # mask and renormalize
            p = p_logits * valid_mask
            if p.sum() > 0:
                p /= p.sum()
            else:
                p = valid_mask / valid_mask.sum()

            self.Ps[s] = p
            self.Vs[s] = valid_mask
            self.Ns[s] = 0
            return -v

        # --- INTERNAL NODE: pick best UCB ---
        valid_mask = self.Vs[s]
        fresh = self.game.getValidMoves(canon, 1)
        if not np.array_equal(fresh, valid_mask):
            print("[SEARCH] valid‐moves changed, updating cache")
            self.Vs[s] = fresh
            valid_mask = fresh

        cur_best, best_a = -float('inf'), -1
        for a in range(self.game.getActionSize()):
            if not valid_mask[a]:
                continue
            if (s, a) in self.Qsa:
                u = ( self.Qsa[(s,a)]
                    + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) 
                                    / (1 + self.Nsa[(s,a)]) )
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

            if u > cur_best:
                cur_best, best_a = u, a

        # --- Expand that move ---
        next_raw, next_player = self.game.getNextState(canon, 1, best_a)
        next_canon      = self.game.getCanonicalForm(next_raw, next_player)
        print(f"{'  ' * (len(path)+1)}")

        # --- Recurse ---
        v = self.search(next_canon, 1, new_path)

        # --- Back-up value ---
        if (s, best_a) in self.Qsa:
            self.Qsa[(s,best_a)] = ((self.Nsa[(s,best_a)] * self.Qsa[(s,best_a)] + v)
                                    / (self.Nsa[(s,best_a)] + 1))
            self.Nsa[(s,best_a)] += 1
        else:
            self.Qsa[(s,best_a)] = v
            self.Nsa[(s,best_a)] = 1
        self.Ns[s] += 1

        return -v

