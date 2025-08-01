from typing import List, Tuple
import numpy as np

from board import Board
from enums import GameState
from numpy.typing import NDArray
import copy
from board import ACTION_SPACE_SIZE
from symmetries import MySymmetry, Rotate60, ReflectVert
from typing import Sequence

USE_SYMMETRIES = True
EXTRA_ACTION = 1

class GameWrapper:
    """
    A simple wrapper that adapts our Board class to the interface expected by Coach.
    """

    def getInitBoard(self, expansions: bool = False) -> Board:
        if expansions:
            return Board("Base+MLP")
        return Board()

    def getNextState(self, board: Board, player: int, action: int, move_str: str = "") -> Tuple[Board,int]:
        # decode against the “real” board, not the canonical one
        if move_str == "":
            move_str = board.decode_move_index(action)

        # 1) Get the binary valid-move mask from the engine:
        valid_mask = self.getValidMoves(board)

        # 2) Assert the chosen action index is 1 in that mask:
        if valid_mask[action] == 0:
            raise ValueError(
                f"Illegal action index {action} "
                f"for player={player} on board {board}"
        )

        print("  >>> getNextState: before play:", board, "action index:", action)
        new = copy.deepcopy(board)
        new.play(move_str)
        print("  >>> getNextState: after play:", new, "next_player:", 1-player)
        return new, 1-player

    def getGameEnded(self, board: Board, player: int) -> float:
        """
        Returns a numerical outcome for the game:
            0 if game is not ended,
            0.5 for a draw,
            1 if the player wins,
           -1 if the player loses.
        """
        if board.state in [GameState.NOT_STARTED, GameState.IN_PROGRESS]:
            return 0.0
        elif board.state == GameState.DRAW:
            return -1e-2  # small bias
        elif board.state == GameState.WHITE_WINS:
            return 1.0 if player == 1 else -1.0
        elif board.state == GameState.BLACK_WINS:
            return 1.0 if player == 0 else -1.0
        return 0.0

    def getSymmetries(
        self, 
        board: Board, 
        pi: NDArray[np.float64]
    ) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        syms: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for t in self.symmetry_transforms():
            b2 = t.apply_to_board(board)
            pi2: NDArray[np.float64] = np.zeros_like(pi)
            # shuffle the probabilities through the action permutation
            for i in range(pi.shape[0]):
                pi2[t.apply_to_action(i)] = pi[i]
            syms.append((b2.encode_board(grid_size=26).astype(np.float64), pi2))
        return syms

    def symmetry_transforms(self) -> Sequence[MySymmetry]:
        return [Rotate60(i) for i in range(6)] + [ReflectVert()] + [Rotate60(i, after_reflection=True) for i in range(6)]

    def getActionSize(self) -> int:
        action_space_size = ACTION_SPACE_SIZE
        return int(action_space_size + 1)  # +1 for the pass move.

    def getValidMoves(self, board: Board) -> NDArray[np.float64]:
        """
        Returns a binary vector of length getActionSize() indicating which moves
        in the fixed tile-relative action space are valid.
        
        For each move string produced by board.valid_moves (a semicolon-separated string),
        we use our encode_move_string function to map it to an index.
        If no moves are valid, we mark the "pass" move as valid.
        """
        action_size = self.getActionSize()
        valid = np.zeros(action_size, dtype=np.float64)
        valid_moves = board.valid_moves_complete().split(";")  
        print("[GET VALID MOVES] Valid moves according to the engine: ", valid_moves)
        # Iterate over each valid move and set the corresponding index to 1.0.
        for move in valid_moves:
            if move:
                try:
                    idx = board.encode_move_string(move, simple=False)
                    valid[idx] = 1.0
                except Exception as e:
                    print(f"Warning: Unable to encode move '{move}': {e}")
        # If no move was marked valid, mark the pass move (the last index) as valid.
        if np.sum(valid) == 0:
            valid[-1] = 1.0
        return valid
    
    def getSynonymSpace(self, board: Board) -> NDArray[np.float64]:
        """
        Returns a binary vector of length getActionSize() indicating with the same id (the first synonym's index) the valid moves that are synonyms.
        """
        valid_moves = board._get_valid_moves()
        synonym_vector = np.zeros(self.getActionSize(), dtype=np.float64)
        for move in valid_moves:
            synonyms = board.stringify_move(move, True).split(";")
            id = board.encode_move_string(synonyms[0], simple=False)
            for synonym in synonyms:
                idx = board.encode_move_string(synonym, simple=False)
                if synonym_vector[idx] != 0:
                    raise ValueError(
                        f"Synonym {synonym}->{id} already has an ID: {synonym_vector[idx]}"
                    )
                synonym_vector[idx] = id
        return synonym_vector