import pytest
from enums import PlayerColor, GameState, GameType, Direction

class TestPlayerColor():
  def test_code(self):
    assert PlayerColor.WHITE.code == "w"
    assert PlayerColor.BLACK.code == "b"

  def test_opposite(self):
    assert PlayerColor.WHITE.opposite == PlayerColor.BLACK
    assert PlayerColor.BLACK.opposite == PlayerColor.WHITE

class TestGameState():
  def test_parse(self):
    assert GameState.parse("NotStarted") == GameState.NOT_STARTED
    assert GameState.parse("InProgress") == GameState.IN_PROGRESS
    assert GameState.parse("Draw") == GameState.DRAW
    assert GameState.parse("WhiteWins") == GameState.WHITE_WINS
    assert GameState.parse("BlackWins") == GameState.BLACK_WINS
    assert GameState.parse("") == GameState.NOT_STARTED

class TestGameType():
  def test_parse(self):
    assert GameType.parse("") == GameType.BASE
    assert GameType.parse("Base") == GameType.BASE
    assert GameType.parse("Base+M") == (GameType.BASE | GameType.M)
    assert GameType.parse("Base+L") == (GameType.BASE | GameType.L)
    assert GameType.parse("Base+P") == (GameType.BASE | GameType.P)
    assert GameType.parse("Base+ML") == (GameType.BASE | GameType.M | GameType.L)
    assert GameType.parse("Base+MP") == (GameType.BASE | GameType.M | GameType.P)
    assert GameType.parse("Base+LM") == (GameType.BASE | GameType.M | GameType.L)
    assert GameType.parse("Base+LP") == (GameType.BASE | GameType.L | GameType.P)
    assert GameType.parse("Base+PM") == (GameType.BASE | GameType.M | GameType.P)
    assert GameType.parse("Base+PL") == (GameType.BASE | GameType.L | GameType.P)
    assert GameType.parse("Base+MLP") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+MPL") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+LMP") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+LPM") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+PML") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+PLM") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    with pytest.raises(ValueError):
      GameType.parse("Invalid")
    with pytest.raises(ValueError):
      GameType.parse("Base+Invalid")
    with pytest.raises(ValueError):
      GameType.parse("M")
    with pytest.raises(ValueError):
      GameType.parse("L")
    with pytest.raises(ValueError):
      GameType.parse("P")

  def test_name(self):
    assert GameType.BASE.name == "Base"
    assert GameType.M.name == "M"
    assert GameType.L.name == "L"
    assert GameType.P.name == "P"

  def test_str(self):
    assert str(GameType.BASE) == "Base"
    assert str(GameType.BASE | GameType.M) == "Base+M"
    assert str(GameType.BASE | GameType.L) == "Base+L"
    assert str(GameType.BASE | GameType.P) == "Base+P"
    assert str(GameType.BASE | GameType.M | GameType.L) == "Base+ML"
    assert str(GameType.BASE | GameType.M | GameType.P) == "Base+MP"
    assert str(GameType.BASE | GameType.L | GameType.M) == "Base+ML"
    assert str(GameType.BASE | GameType.L | GameType.P) == "Base+LP"
    assert str(GameType.BASE | GameType.P | GameType.M) == "Base+MP"
    assert str(GameType.BASE | GameType.P | GameType.L) == "Base+LP"
    assert str(GameType.BASE | GameType.M | GameType.L | GameType.P) == "Base+MLP"
    assert str(GameType.BASE | GameType.M | GameType.P | GameType.L) == "Base+MLP"
    assert str(GameType.BASE | GameType.L | GameType.M | GameType.P) == "Base+MLP"
    assert str(GameType.BASE | GameType.L | GameType.P | GameType.M) == "Base+MLP"
    assert str(GameType.BASE | GameType.P | GameType.M | GameType.L) == "Base+MLP"
    assert str(GameType.BASE | GameType.P | GameType.L | GameType.M) == "Base+MLP"

class TestDirection():
  def test_flat(self):
    directions = Direction.flat()
    assert Direction.RIGHT in directions
    assert Direction.UP_RIGHT in directions
    assert Direction.UP_LEFT in directions
    assert Direction.LEFT in directions
    assert Direction.DOWN_LEFT in directions
    assert Direction.DOWN_RIGHT in directions
    assert Direction.BELOW not in directions
    assert Direction.ABOVE not in directions

  def test_flat_left(self):
    directions = Direction.flat_left()
    assert Direction.RIGHT not in directions
    assert Direction.UP_RIGHT not in directions
    assert Direction.UP_LEFT in directions
    assert Direction.LEFT in directions
    assert Direction.DOWN_LEFT in directions
    assert Direction.DOWN_RIGHT not in directions
    assert Direction.BELOW not in directions
    assert Direction.ABOVE not in directions

  def test_flat_right(self):
    directions = Direction.flat_right()
    assert Direction.RIGHT in directions
    assert Direction.UP_RIGHT in directions
    assert Direction.UP_LEFT not in directions
    assert Direction.LEFT not in directions
    assert Direction.DOWN_LEFT not in directions
    assert Direction.DOWN_RIGHT in directions
    assert Direction.BELOW not in directions
    assert Direction.ABOVE not in directions

  def test_str(self):
    assert str(Direction.RIGHT) == "-"
    assert str(Direction.UP_RIGHT) == "/"
    assert str(Direction.UP_LEFT) == "\\"
    assert str(Direction.LEFT) == "-"
    assert str(Direction.DOWN_LEFT) == "/"
    assert str(Direction.DOWN_RIGHT) == "\\"
    assert str(Direction.BELOW) == ""
    assert str(Direction.ABOVE) == ""

  def test_opposite(self):
    assert Direction.RIGHT.opposite == Direction.LEFT
    assert Direction.UP_RIGHT.opposite == Direction.DOWN_LEFT
    assert Direction.UP_LEFT.opposite == Direction.DOWN_RIGHT
    assert Direction.LEFT.opposite == Direction.RIGHT
    assert Direction.DOWN_LEFT.opposite == Direction.UP_RIGHT
    assert Direction.DOWN_RIGHT.opposite == Direction.UP_LEFT
    assert Direction.BELOW.opposite == Direction.ABOVE
    assert Direction.ABOVE.opposite == Direction.BELOW

  def test_left_of(self):
    assert Direction.RIGHT.left_of == Direction.UP_RIGHT
    assert Direction.UP_RIGHT.left_of == Direction.UP_LEFT
    assert Direction.UP_LEFT.left_of == Direction.LEFT
    assert Direction.LEFT.left_of == Direction.DOWN_LEFT
    assert Direction.DOWN_LEFT.left_of == Direction.DOWN_RIGHT
    assert Direction.DOWN_RIGHT.left_of == Direction.RIGHT
    assert Direction.BELOW.left_of == Direction.BELOW
    assert Direction.ABOVE.left_of == Direction.ABOVE

  def test_right_of(self):
    assert Direction.RIGHT.right_of == Direction.DOWN_RIGHT
    assert Direction.UP_RIGHT.right_of == Direction.RIGHT
    assert Direction.UP_LEFT.right_of == Direction.UP_RIGHT
    assert Direction.LEFT.right_of == Direction.UP_LEFT
    assert Direction.DOWN_LEFT.right_of == Direction.LEFT
    assert Direction.DOWN_RIGHT.right_of == Direction.DOWN_LEFT
    assert Direction.BELOW.right_of == Direction.BELOW
    assert Direction.ABOVE.right_of == Direction.ABOVE

  def test_delta_index(self):
    assert Direction.RIGHT.delta_index == 0
    assert Direction.UP_RIGHT.delta_index == 1
    assert Direction.UP_LEFT.delta_index == 2
    assert Direction.LEFT.delta_index == 3
    assert Direction.DOWN_LEFT.delta_index == 4
    assert Direction.DOWN_RIGHT.delta_index == 5
    assert Direction.BELOW.delta_index == 6
    assert Direction.ABOVE.delta_index == 7

  def test_is_right(self):
    assert Direction.RIGHT.is_right
    assert Direction.UP_RIGHT.is_right
    assert Direction.DOWN_RIGHT.is_right
    assert not Direction.UP_LEFT.is_right
    assert not Direction.LEFT.is_right
    assert not Direction.DOWN_LEFT.is_right
    assert not Direction.BELOW.is_right
    assert not Direction.ABOVE.is_right

  def test_is_left(self):
    assert not Direction.RIGHT.is_left
    assert not Direction.UP_RIGHT.is_left
    assert not Direction.DOWN_RIGHT.is_left
    assert Direction.UP_LEFT.is_left
    assert Direction.LEFT.is_left
    assert Direction.DOWN_LEFT.is_left
    assert not Direction.BELOW.is_left
    assert not Direction.ABOVE.is_left

if __name__ == '__main__':
  pytest.main()
