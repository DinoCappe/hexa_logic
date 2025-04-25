from typing import Any
from enum import IntEnum

from dictionaries import DIRECTION_MAP_PREFIX, DIRECTION_MAP_SUFFIX

class dotdict(dict[str, Any]):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'dotdict' object has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
    
    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'dotdict' object has no attribute '{key}'")
        
class Direction(IntEnum):
    TOP_RIGHT     = 0
    RIGHT         = 1
    BOTTOM_RIGHT  = 2
    BOTTOM_LEFT   = 3
    LEFT          = 4
    TOP_LEFT      = 5

    @staticmethod
    def from_symbol(sym: str, *, position: str) -> "Direction":
        """
        :param sym: one of "/", "-", "\\"  (must not be "")
        :param position: either "prefix" or "suffix"
        """
        if position == "suffix":
            idx = DIRECTION_MAP_SUFFIX.get(sym)
        elif position == "prefix":
            idx = DIRECTION_MAP_PREFIX.get(sym)
        else:
            raise ValueError(f"position must be 'prefix' or 'suffix', got {position!r}")
        if idx is None:
            raise ValueError(f"Unknown direction symbol '{sym}' for {position}")
        return Direction(idx)