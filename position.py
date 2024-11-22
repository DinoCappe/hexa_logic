from typing import Final

class Position():
  """
  Tile position.
  """
  def __init__(self, q: int, r: int):
    self.q: Final[int] = q
    self.r: Final[int] = r

  def __hash__(self) -> int:
    return hash((self.q, self.r))

  def __eq__(self, value: object) -> bool:
    return self is value or isinstance(value, Position) and self.q == value.q and self.r == value.r

  def __add__(self, other: object):
    return Position(self.q + other.q, self.r + other.r) if isinstance(other, Position) else NotImplemented
    
  def __sub__(self, other: object):
    return Position(self.q - other.q, self.r - other.r) if isinstance(other, Position) else NotImplemented
