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
        print("  >>> getCanonicalForm: incoming board:", board, "player:", player)
        if player == 1:
            canon = board
        else:
            canon = board.invert_colors()
        print("  >>> getCanonicalForm: outgoing canon:", canon)
        return canon

    def getNextState(self, board: Board, player: int, action: int) -> Tuple[Board,int]:
        # decode against the “real” board, not the canonical one
        move_str = board.decode_move_index(player, action)

        # ----- NEW DEBUGGING ASSERTION -----
        valid = board.valid_moves.split(";")
        assert move_str in valid, (
            f"DECODED move_str `{move_str}` not in board.valid_moves!\n"
            f"  board was: {board}\n"
            f"  valid_moves: {valid}"
        )
        # -----------------------------------
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

        # first 6 rotations by 60°
        for _ in range(6):
            # flatten the spatial π back to 1d and tack on the extra slot
            full_pi = np.concatenate((cur_pi.ravel(), extra))

            # append *copies* of both arrays
            sym_list.append((cur_board.copy(), full_pi.copy()))

            # rotate in preparation for the next one
            cur_board = board.rotate_board_60(cur_board)
            cur_pi    = board.rotate_pi_60(cur_pi)

        #––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # reflect once
        cur_board = board.reflect_board_vertically(cur_board)
        cur_pi    = board.reflect_pi_vertically(cur_pi)

        # then 6 more rotations of that reflected version
        for _ in range(6):
            full_pi = np.concatenate((cur_pi.ravel(), extra))
            sym_list.append((cur_board.copy(), full_pi.copy()))

            cur_board = board.rotate_board_60(cur_board)
            cur_pi    = board.rotate_pi_60(cur_pi)

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