import pytest
from enums import PlayerColor, BugType, Direction
from game import Position, Bug, Move

class TestPosition():
  def test_init(self):
    pos = Position(1, 2)
    assert pos.q == 1
    assert pos.r == 2

  def test_str(self):
    pos = Position(1, 2)
    assert str(pos) == "(1, 2)"

  def test_hash(self):
    pos1 = Position(1, 2)
    pos2 = Position(1, 2)
    pos3 = Position(2, 3)
    assert hash(pos1) == hash(pos2)
    assert hash(pos1) != hash(pos3)

  def test_eq(self):
    pos1 = Position(1, 2)
    pos2 = Position(1, 2)
    pos3 = Position(2, 3)
    assert pos1 == pos2
    assert pos1 != pos3

  def test_add_sub(self):
    pos1 = Position(1, 2)
    pos2 = Position(3, 4)
    pos3 = Position(4, 6)
    assert pos1 + pos2 == pos3
    assert pos3 - pos2 == pos1

class TestBug():
  def test_parse_valid(self):
    bug = Bug.parse("wQ")
    assert bug.color == PlayerColor.WHITE
    assert bug.type == BugType.QUEEN_BEE
    assert bug.id == 0

  def test_parse_invalid(self):
    with pytest.raises(ValueError):
      Bug.parse("INVALID")

  def test_init(self):
    bug = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    assert bug.color == PlayerColor.WHITE
    assert bug.type == BugType.QUEEN_BEE
    assert bug.id == 0

  def test_str(self):
    bug1 = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    bug2 = Bug(PlayerColor.BLACK, BugType.BEETLE, 3)
    assert str(bug1) == "wQ"
    assert str(bug2) == "bB3"

  def test_hash(self):
    bug1 = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    bug2 = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE, 0)
    bug3 = Bug(PlayerColor.BLACK, BugType.BEETLE, 1)
    assert hash(bug1) == hash(bug2)
    assert hash(bug1) != hash(bug3)

  def test_eq(self):
    bug1 = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    bug2 = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE, 0)
    bug3 = Bug(PlayerColor.BLACK, BugType.BEETLE, 1)
    assert bug1 == bug2
    assert bug1 != bug3

class TestMove():
  def test_stringify(self):
    bug1 = Bug(PlayerColor.WHITE, BugType.SPIDER, 1)
    move_str = Move.stringify(bug1)
    assert move_str == "wS1"

    bug2 = Bug(PlayerColor.BLACK, BugType.SOLDIER_ANT, 1)
    move_str = Move.stringify(bug2, bug1, Direction.RIGHT)
    assert move_str == "bA1 wS1-"

    bug3 = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    move_str = Move.stringify(bug3, bug1, Direction.LEFT)
    assert move_str == "wQ -wS1"

  def test_init(self):
    bug = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    origin = Position(1, 1)
    destination = Position(1, 0)
    move = Move(bug, origin, destination)
    assert move.bug == bug
    assert move.origin == origin
    assert move.destination == destination

  def test_str(self):
    bug = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    origin = Position(1, 1)
    destination = Position(1, 0)
    move = Move(bug, origin, destination)
    assert str(move) == "wQ, (1, 1), (1, 0)"

  def test_hash(self):
    bug = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    origin = Position(1, 1)
    destination = Position(1, 0)
    move1 = Move(bug, origin, destination)
    move2 = Move(bug, origin, destination)
    move3 = Move(bug, origin, Position(0, 1))
    assert hash(move1) == hash(move2)
    assert hash(move1) != hash(move3)

  def test_eq(self):
    bug = Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)
    origin = Position(1, 1)
    destination = Position(1, 0)
    move1 = Move(bug, origin, destination)
    move2 = Move(bug, origin, destination)
    move3 = Move(bug, origin, Position(0, 1))
    assert move1 == move2
    assert move1 != move3

if __name__ == "__main__":
  pytest.main()
