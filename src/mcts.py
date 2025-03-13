import math
import numpy as np
from typing import Dict, Tuple
from numpy.typing import NDArray
from board import Board
from ai import Brain
from coach import GameWrapper
from HiveNNet import NNetWrapper
from utils import dotdict
from enums import PlayerColor

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
        """
        Performs numMCTSSims simulations starting from canonicalBoard and returns
        a probability distribution over actions (of length getActionSize()).
        Temperature temp controls exploration (temp=0 -> deterministic).
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
        if temp == 0:
            bestAs = np.argwhere(np.array(counts) == np.max(counts)).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(len(counts), dtype=np.float64)
            probs[bestA] = 1.0
            return probs
        counts = [x ** (1.0 / temp) for x in counts]
        total = float(sum(counts))
        probs = [x / total for x in counts]
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
        
        move_str = board.decode_move_index(best_action_index, player)
        return move_str

    def search(self, canonicalBoard: Board) -> float:
        """
        Recursively searches from the given canonicalBoard state.
        Returns the negative value of the state.
        """
        s = self.game.stringRepresentation(canonicalBoard)

        # If terminal state, return outcome.
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
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
        cur_best = -float('inf')
        best_a = -1
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                if u > cur_best:
                    cur_best = u
                    best_a = a

        a = best_a
        next_board, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_board = self.game.getCanonicalForm(next_board, next_player)
        v = self.search(next_board)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v
