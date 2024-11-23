from typing import Final
from enums import PlayerColor, BugType, Direction
import re

class Position():
  """
  Tile position.
  """
  def __init__(self, q: int, r: int):
    self.q: Final[int] = q
    self.r: Final[int] = r

  def __str__(self) -> str:
    return f"({self.q}, {self.r})"

  def __hash__(self) -> int:
    return hash((self.q, self.r))

  def __eq__(self, value: object) -> bool:
    return self is value or isinstance(value, Position) and self.q == value.q and self.r == value.r

  def __add__(self, other: object):
    return Position(self.q + other.q, self.r + other.r) if isinstance(other, Position) else NotImplemented
    
  def __sub__(self, other: object):
    return Position(self.q - other.q, self.r - other.r) if isinstance(other, Position) else NotImplemented

class Bug():
  """
  Bug piece.
  """
  COLORS: Final[dict[str, PlayerColor]] = {color.code: color for color in PlayerColor}
  """
  Color code map.
  """
  REGEX: Final[str] = f"({"|".join(COLORS.keys())})({"|".join(BugType)})(1|2|3)?"
  """
  Regex to validate BugStrings.
  """

  @classmethod
  def parse(cls, bug: str):
    """
    Parses a BugString.

    :param bug: BugString.
    :type bug: str
    :raises ValueError: If it's not a valid BugString.
    :return: Bug piece.
    :rtype: Bug
    """
    match = re.fullmatch(cls.REGEX, bug)
    if match:
      color, type, id = match.groups()
      return Bug(cls.COLORS[color], BugType(type), int(id or 0))
    raise ValueError(f"'{bug}' is not a valid BugString")

  def __init__(self, color: PlayerColor, bug_type: BugType, bug_id: int = 0) -> None:
    self.color: Final[PlayerColor] = color
    self.type: Final[BugType] = bug_type
    self.id: Final[int] = bug_id

  def __hash__(self) -> int:
    return hash(str(self))
  
  def __eq__(self, value: object) -> bool:
    return self is value or isinstance(value, Bug) and self.color is value.color and self.type is value.type and self.id == value.id

  def __str__(self) -> str:
    return f"{self.color.code}{self.type}{self.id if self.id else ""}"

class Move():
  """
  Move.
  """
  PASS: Final[str] = "pass"
  """
  Pass move.
  """
  REGEX = f"({Bug.REGEX})( ?({"|".join(f"\\{d}" for d in Direction.flat_left())})?({Bug.REGEX})({"|".join(f"\\{d}" for d in Direction.flat_right())})?)?"
  """
  MoveString regex.
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
  
  def __str__(self) -> str:
    return f"{self.bug}, {self.origin}, {self.destination}"
