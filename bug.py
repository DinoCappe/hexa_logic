from typing import Final
from enums import PlayerColor, BugType
import re

class Bug():
  """
  Bug piece.
  """
  colors: Final[dict[str, PlayerColor]] = {color.code: color for color in PlayerColor}
  """
  Color code map.
  """
  regex: Final[str] = f"({"|".join(colors.keys())})({"|".join(BugType)})(1|2|3)?"
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
    match = re.fullmatch(cls.regex, bug)
    if match:
      color, type, id = match.groups()
      return Bug(cls.colors[color], BugType(type), int(id or 0))
    raise ValueError(f"'{bug}' is not a valid BugString")

  def __init__(self, color: PlayerColor, bug_type: BugType, bug_id: int) -> None:
    self.color: Final[PlayerColor] = color
    self.type: Final[BugType] = bug_type
    self.id: Final[int] = bug_id

  def __hash__(self) -> int:
    return hash(str(self))
  
  def __eq__(self, value: object) -> bool:
    return self is value or isinstance(value, Bug) and self.color is value.color and self.type is value.type and self.id == value.id

  def __str__(self) -> str:
    return f"{self.color.code}{self.type}{self.id if self.id else ""}"

