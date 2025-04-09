from typing import List, Tuple
import numpy as np

from board import Board
from enums import GameState
from numpy.typing import NDArray
import copy
from board import ACTION_SPACE_SIZE

USE_SYMMETRIES = True
EXTRA_ACTION = 1

class GameWrapper:
    """
    A simple wrapper that adapts our Board class to the interface expected by Coach.
    """

    def getInitBoard(self) -> Board:
        return Board()
    
    def getCanonicalForm(self, board: Board, player: int) -> Board:
        if player == 1:
            return board
        else:
            canon = board.invert_colors()
            canon.turn = canon.turn - 1  
            return canon

    def getNextState(self, board: Board, player: int, action: int) -> Tuple[Board, int]:
        """
        Applies the move corresponding to the action index and returns a new board state.
        """
        move_str = board.decode_move_index(player, action)
        print("Best move string: ", move_str)
        
        new_board = copy.deepcopy(board)    # infinite recursion issue
        new_board.play(move_str)
        print("New board after playing move: ", new_board)
        
        return new_board, -player

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
            # TODO: Make sense 0.5? Should be 0.0 for both players?
            return 0.5
        elif board.state == GameState.WHITE_WINS:
            return 1.0 if player == 1 else -1.0
        elif board.state == GameState.BLACK_WINS:
            return 1.0 if player == 0 else -1.0
        return 0.0

    def getSymmetries(self, board: Board, pi: NDArray[np.float64]) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Return a list of (board, pi) pairs for all 12 symmetries of the board.
        The board is represented as a numpy array (obtained from board.encode_board),
        and pi is a policy vector that is assumed to be spatially encoded.
        
        If symmetries are disabled or the policy vector doesn't match expectations,
        simply return the original board encoding and pi.
        """

        if not USE_SYMMETRIES:
            return [(board.encode_board(grid_size=14).astype(np.float64), pi)]

        np_board = board.encode_board(grid_size=14) # shape (channels, 14, 14)
        # print("[GET SYMMETRIES] np board: ", np_board)
        
        expected_length = ACTION_SPACE_SIZE + EXTRA_ACTION
        print("Expected length: ", expected_length)
        print("Pi length: ", pi.shape[0])
        if pi.shape[0] != expected_length:
            return [(np_board.astype(np.float64), pi)]
        
        # Separate the extra action (if any) from the spatial part.
        pi_board = np.reshape(pi[:-EXTRA_ACTION], ACTION_SPACE_SIZE)
        extra = pi[-EXTRA_ACTION:]
        
        sym_list: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        cur_board = np.asarray(np_board, dtype=np.float64)  # Ensure float64
        cur_pi = np.asarray(pi_board, dtype=np.float64)
        extra = np.asarray(extra, dtype=np.float64)

        # First 6 rotations (each by 60 degrees)
        for _i in range(6):
            combined = np.concatenate((cur_pi.ravel(), extra))
            sym_list.append((cur_board, combined))

            cur_board = board.rotate_board_60(cur_board).astype(np.float64)
            cur_pi = board.rotate_pi_60(cur_pi).astype(np.float64)

        # Reflect vertically.
        cur_board = board.reflect_board_vertically(cur_board).astype(np.float64)
        cur_pi = board.reflect_pi_vertically(cur_pi).astype(np.float64)

        # Next 6 rotations of the reflected board.
        for _i in range(6):
            combined = np.concatenate((cur_pi.ravel(), extra))
            sym_list.append((cur_board, combined))
            cur_board = board.rotate_board_60(cur_board).astype(np.float64)
            cur_pi = board.rotate_pi_60(cur_pi).astype(np.float64)

        print("[GET SYMMETRIES] Sym list length: ", len(sym_list))
        return sym_list

    def getActionSize(self) -> int:
        action_space_size = ACTION_SPACE_SIZE
        return int(action_space_size + 1)  # +1 for the pass move.

    def getValidMoves(self, board: Board, player: int) -> NDArray[np.float64]:
        """
        Returns a binary vector of length getActionSize() indicating which moves
        in the fixed tile-relative action space are valid.
        
        For each move string produced by board.valid_moves (a semicolon-separated string),
        we use our encode_move_string function to map it to an index.
        If no moves are valid, we mark the "pass" move as valid.
        """
        action_size = self.getActionSize()
        valid = np.zeros(action_size, dtype=np.float64)
        valid_moves = board.valid_moves.split(";")
        print("[GET VALID MOVES] Valid moves according to the engine: ", valid_moves)
        # Iterate over each valid move and set the corresponding index to 1.0.
        for move in valid_moves:
            if move:
                try:
                    idx = board.encode_move_string(move, player, simple=False)
                    valid[idx] = 1.0
                except Exception as e:
                    print(f"Warning: Unable to encode move '{move}': {e}")
        # If no move was marked valid, mark the pass move (the last index) as valid.
        if np.sum(valid) == 0:
            valid[-1] = 1.0
        return valid