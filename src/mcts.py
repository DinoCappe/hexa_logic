import math
import numpy as np
from typing import Dict, Tuple
from numpy.typing import NDArray
from board import Board
from ai import Brain
from gameWrapper import GameWrapper
from HiveNNet import NNetWrapper
from utils import dotdict
from enums import PlayerColor
# from copy import deepcopy

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
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
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
        It uses getActionProb with temperature 0 for deterministic selection.
        """
        # Get probability distribution from MCTS (deterministic with temp=0)
        probs = self.getActionProb(board, temp=0)
        best_action_index = int(np.argmax(probs))
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0
        
        move_str = board.decode_move_index(player, best_action_index)
        return move_str

    def search(self, canonicalBoard: Board) -> float:
        """
        Recursively searches from the given canonicalBoard state.
        Returns the negative value of the state.
        """
        s = self.game.stringRepresentation(canonicalBoard)
        print("[SEARCH] Using board state:", canonicalBoard)

        # If terminal state, return outcome.
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            #print("[SEARCH] Game outcome from getGameEnded:", self.Es[s])
        if self.Es[s] != 0:
            #print("[SEARCH] Terminal state detected. Outcome:", self.Es[s])
            return -self.Es[s]

        # If state is not in cache, it's a leaf.
        if s not in self.Ps:
            # Query the NN for policy and value.
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # mask invalid moves
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps
            else:
                # If all valid moves were masked, use uniform probabilities.
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # Otherwise, choose the action with highest UCB.
        valids = self.Vs[s]
        print("[SEARCH] Valid moves for a state that's been seen before: ", canonicalBoard.valid_moves.split(";"))

        cur_best = -float('inf')
        best_a = -1
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                print(f"[SEARCH] Action {a}: UCB value = {u}, visits = {self.Nsa.get((s, a), 0)}")
                if u > cur_best:
                    cur_best = u
                    best_a = a

        a = best_a
        print("Best move index: ", a)
        player = 1 if canonicalBoard.current_player_color == PlayerColor.WHITE else 0
        next_board, next_player = self.game.getNextState(canonicalBoard, player, a)
        #print("[SEARCH] Next board string:", self.game.stringRepresentation(next_board))
        if canonicalBoard.turn % 2 == 1:
            next_board = self.game.getCanonicalForm(next_board, next_player)
        v = self.search(next_board)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        #print("[SEARCH] Updated state:", s, "visit count:", self.Ns[s], "Qsa for action", a, "=", self.Qsa[(s, a)])
        return -v
