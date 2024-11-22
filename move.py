from typing import Final
from enums import Direction
from position import Position
from bug import Bug

class Move():
  """
  Move.
  """
  PASS: Final[str] = "pass"
  """
  Pass move.
  """

  @classmethod
  def stringify(cls, moved: Bug, relative: Bug | None = None, direction: Direction | None = None) -> str:
    """
    Converts the data for a move to the corresponding MoveString.

    :param moved: Bug piece moved.
    :type moved: Bug
    :param relative: Bug piece relative to which the other bug piece is moved, defaults to None.
    :type relative: Bug | None, optional
    :param direction: Direction of the destination tile with respect to the relative bug piece, defaults to None.
    :type direction: Direction | None, optional
    :return: MoveString.
    :rtype: str
    """
    return f"{moved} {direction if direction.is_left else ""}{relative}{direction if direction.is_right else ""}" if relative and direction else f"{moved}"

  def __init__(self, bug: Bug, origin: Position | None, destination: Position) -> None:
    self.bug: Final[Bug] = bug
    self.origin: Final[Position | None] = origin
    self.destination: Final[Position] = destination

  def __hash__(self) -> int:
    return hash((self.bug, self.origin, self.destination))

  def __eq__(self, value: object) -> bool:
    return self is value or isinstance(value, Move) and self.bug == value.bug and self.origin == value.origin and self.destination == value.destination
