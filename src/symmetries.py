from abc import ABC, abstractmethod
from board import Board
from dictionaries import DIRECTION_MAP_PREFIX, DIRECTION_MAP_SUFFIX
from game import Move
import re
from typing import List
from board import ACTION_SPACE_SIZE
from utils import Direction

FLIP_DIRECTION_PERMUTATION = [2, 1, 0, 5, 4, 3, 6]
PREFIX_SYMBOLS = {v:k for k, v in DIRECTION_MAP_PREFIX.items()}
SUFFIX_SYMBOLS = {v:k for k, v in DIRECTION_MAP_SUFFIX.items()}

class MySymmetry(ABC):
    
    def __init__(self):
        init_board = Board()
        white_player_id = 1

        self.action_list: List[str] = []
        for idx in range(self.getActionSize()):
            mv = init_board.decode_move_index(white_player_id, idx)
            self.action_list.append(mv)

        self.action_to_index = { mv: i for i, mv in enumerate(self.action_list) }

    def getActionSize(self) -> int:
        action_space_size = ACTION_SPACE_SIZE
        return int(action_space_size + 1)

    @abstractmethod
    def apply_to_board(self, board: Board) -> Board:
        """Return a new Board object with this symmetry applied."""
        pass

    @abstractmethod
    def apply_to_action(self, action_idx: int) -> int:
        """Return a new move‐string that represents `move_str` under this symmetry."""
        pass

class Rotate60(MySymmetry):
    def __init__(self, steps: int, after_reflection: bool = False):
        super().__init__()
        self.steps = steps % 6
        self.after_reflection = after_reflection
        # pre‐compile for speed
        self._rx = re.compile(Move.REGEX)

    def apply_to_board(self, board: Board) -> Board:
        b = board
        if self.after_reflection:
            b = b.reflect_vertically()
        for _ in range(self.steps):
            b = b.rotate_60()
        return b
    
    def idx_to_symbol(self, idx: int, position: str) -> str:
        """
        Turn a 0–5 direction index back into one of "/", "-", "\\" in
        either the PREFIX or SUFFIX sense.
        """
        if position == "prefix":
            return PREFIX_SYMBOLS[idx]
        elif position == "suffix":
            return SUFFIX_SYMBOLS[idx]
        else:
            raise ValueError("position must be 'prefix' or 'suffix'")

    def apply_to_action(self, action_idx: int) -> int:
        move_str = self.action_list[action_idx]

        # single‐token placement moves don’t move under any symmetry
        # (no space → no neighbor‐direction syntax)
        if " " not in move_str and move_str != Move.PASS:
            return action_idx

        if move_str == Move.PASS:
            return action_idx

        m = self._rx.fullmatch(move_str)
        if not m:
            raise ValueError(f"Bad move string: {move_str}")

        bug_token   = m.group(1)               # e.g. "wQ"
        neigh_arrow = m.group(5) or ""         # prefix indicator, or ""
        target_bug  = m.group(6) or ""         # the second‐bug token, or ""
        drop_arrow  = m.group(9) or ""         # suffix indicator, or ""

        # only parse when nonempty:
        nd = Direction.from_symbol(neigh_arrow, position="prefix").value \
             if neigh_arrow else None
        dd = Direction.from_symbol(drop_arrow, position="suffix").value \
             if drop_arrow else None

        # reflect‐then‐rotate exactly as before, but skip reflect/rotate on None:
        if self.after_reflection and nd is not None:
            nd = FLIP_DIRECTION_PERMUTATION[nd]
        if self.after_reflection and dd is not None:
            dd = FLIP_DIRECTION_PERMUTATION[dd]

        if nd is not None:
            nd = (nd + self.steps) % 6
        if dd is not None:
            dd = (dd + self.steps) % 6

        # rebuild the move‐string:
        new_move = bug_token
        if nd is not None:
            new_move += self.idx_to_symbol(nd, position="prefix")
        new_move += target_bug
        if dd is not None:
            new_move += self.idx_to_symbol(dd, position="suffix")

        return self.action_list.index(new_move)

class ReflectVert(MySymmetry):
    def __init__(self):
        super().__init__()
        self._inner = Rotate60(steps=0, after_reflection=True)

    def apply_to_board(self, board: Board) -> Board:
        return self._inner.apply_to_board(board)

    def apply_to_action(self, action_idx: int) -> int:
        return self._inner.apply_to_action(action_idx)
