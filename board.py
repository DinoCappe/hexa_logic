from typing import Final, Set
from enums import GameType, GameState, PlayerColor, BugType, Direction
from position import Position
from bug import Bug
from move import Move
import re

class Board():
  """
  Game Board.
  """
  ORIGIN: Final[Position] = Position(0, 0)
  """
  Position of the first piece played.
  """
  NEIGHBOR_DELTAS: Final[tuple[Position, Position, Position, Position, Position, Position]] = (
    Position(1, 0), # Right
    Position(1, -1), # Up right
    Position(0, -1), # Up left
    Position(-1, 0), # Left
    Position(-1, 1), # Down left
    Position(0, 1), # Down right
  )
  """
  Offsets of every neighboring tile in each flat direction.
  """

  def __init__(self, gamestring: str = "") -> None:
    """
    Game Board instantation.

    :param gamestring: GameString, defaults to "".
    :type gamestring: str, optional
    """
    type, state, turn, moves = self._parse_gamestring(gamestring)
    self.type: Final[GameType] = type
    self.state: GameState = state
    self.turn: int = turn
    self.moves: list[str] = moves # TODO: Need to implement a history of the board, most likely the moves played in order, to be able to undo them.
    self.last_piece_moved: Bug | None = None # TODO: Add functionality. Might not be needed, as it could be better to simply check the last move.
    self._pos_to_bug: dict[Position, list[Bug]] = {}
    """
    Map for tile positions on the board and bug pieces placed there (pieces can be stacked).
    """
    self._bug_to_pos: dict[Bug, Position | None] = {}
    """
    Map for bug pieces and their current position.  
    Position is None if the piece has not been played yet.  
    Also serves as a check for valid pieces for the game.
    """
    for color in PlayerColor:
      for expansion in self.type:
        if expansion is GameType.Base:
          self._bug_to_pos[Bug(color, BugType.QUEEN_BEE, 0)] = None
          # Add ids greater than 0 only for bugs with multiple copies.
          for i in range(1, 3):
            self._bug_to_pos[Bug(color, BugType.SPIDER, i)] = None
            self._bug_to_pos[Bug(color, BugType.BEETLE, i)] = None
            self._bug_to_pos[Bug(color, BugType.GRASSHOPPER, i)] = None
            self._bug_to_pos[Bug(color, BugType.SOLDIER_ANT, i)] = None
          self._bug_to_pos[Bug(color, BugType.GRASSHOPPER, 3)] = None
          self._bug_to_pos[Bug(color, BugType.SOLDIER_ANT, 3)] = None
        else:
          self._bug_to_pos[Bug(color, BugType(expansion.name), 0)] = None
    self._valid_moves_cache: Set[Move] | None = None
    # TODO: Most likely a cache for the best move will be needed too.

  def __str__(self) -> str:
    return f"{self.type};{self.state};{self.current_player_color}[{self.current_player_turn}]{';' if len(self.moves) else ''}{';'.join(self.moves)}"

  def _parse_turn(self, turn: str) -> int:
    """
    Parses a TurnString.

    :param turn: TurnString.
    :type turn: str
    :raises ValueError: If it's not a valid TurnString.
    :return: Turn number.
    :rtype: int
    """
    match = re.fullmatch(f"({PlayerColor.WHITE}|{PlayerColor.BLACK})\\[(\\d+)\\]", turn)
    if match:
      color, player_turn = match.groups()
      return 2 * int(player_turn) - 2 + list(PlayerColor).index(PlayerColor(color))
    else:
      raise ValueError(f"'{turn}' is not a valid TurnString")

  def _parse_move(self, move: str) -> str:
    """
    Parses a string representing a move.

    :param moves: Stringified list of moves.
    :type moves: str
    :return: List of moves.
    :rtype: list[str]
    """
    return move

  def _parse_gamestring(self, gamestring: str) -> tuple[GameType, GameState, int, list[str]]:
    """
    Parses a GameString.

    :param gamestring: GameString.
    :type gamestring: str
    :raises TypeError: If it's not a valid GameString.
    :return: Tuple of GameString components, namely GameType, GameState, turn number, and list of moves made so far.
    :rtype: tuple[GameType, GameState, int, list[str]]
    """
    values = gamestring.split(";") if gamestring else ["", "", f"{PlayerColor.WHITE}[1]"]
    if len(values) == 1:
      values += ["", f"{PlayerColor.WHITE}[1]"]
    elif len(values) < 3:
      raise TypeError(f"'{gamestring}' is not a valid GameString")
    type, state, turn, *moves = values
    return GameType.parse(type), GameState.parse(state), self._parse_turn(turn), moves

  def _make_initial_moves(self, moves: list[str]) -> list[str]:
    if self.turn - 1 == len(moves):
      for move in moves:
        self.play(move)
      return moves
    raise ValueError(f"Expected {self.turn - 1} moves but got {len(moves)}")

  def play(self, move: str) -> None:
    self._parse_move(move) # check also if it's valid (legal given the current board state). valid moves should also tell you whether or not you can/must play your queen.
    self.turn += 1
    if self.state is GameState.NOT_STARTED:
      self.state = GameState.IN_PROGRESS
    # Retrieve bug-to-move from self.bug_to_pos
    # Retrieve relative-bug from self.bug_to_pos
    # Retrieve Direction from move, then get the delta from NEIGHBOR_DELTAS
    # Check that bug-to-move can be moved next to relative-bug in the specified Direction
      # Calculate the resulting Position
      # Check the tile state from self.pos_to_bug
    # Move bug-to-move, updating both self.bug_to_pos and self.pos_to_bug
    # Clear _valid_moves_cache (check also if the move is valid initially)
    # Update self.last_piece_moved

  @property
  def current_player_color(self) -> PlayerColor:
    """
    Color of the current player.

    :rtype: PlayerColor
    """
    return list(PlayerColor)[self.turn % len(PlayerColor)]

  @property
  def current_player_turn(self) -> int:
    """
    Turn number of the current player.

    :rtype: int
    """
    return 1 + self.turn // 2

  @property
  def current_player_queen_in_play(self) -> bool:
    """
    Whether the current player's queen bee is in play.

    :rtype: bool
    """
    return bool(self._bug_to_pos[Bug(self.current_player_color, BugType.QUEEN_BEE, 0)])

  @property
  def valid_moves(self) -> str:
    """
    Current possible legal moves in a joined list of MoveStrings.

    :rtype: str
    """
    moves = self.get_valid_moves()
    move_strings: list[str] = []
    for move in moves:
      moved: Bug = move.bug
      relative: Bug | None = None
      direction: Direction | None = None
      dest_bugs = self.bugs_from_pos(move.destination)
      if dest_bugs:
        relative = dest_bugs[-1]
      else:
        for dir in Direction.flat():
          neighbor_bugs = self.bugs_from_pos(self.get_neighbor(move.destination, dir))
          if neighbor_bugs:
            relative = neighbor_bugs[0]
            direction = dir
            break
      move_strings.append(Move.stringify(moved, relative, direction))
    return ";".join(move_strings) or Move.PASS

  @property
  def best_move(self) -> str:
    """
    Current best move as a MoveString.

    :rtype: str
    """
    move = self.get_best_move()
    return self.stringify_move(move) if move else Move.PASS
  
  def stringify_move(self, move: Move) -> str:
    """
    Returns a MoveString from the given move.

    :param move: Move.
    :type move: Move
    :return: MoveString.
    :rtype: str
    """
    moved: Bug = move.bug
    relative: Bug | None = None
    direction: Direction | None = None
    dest_bugs = self.bugs_from_pos(move.destination)
    if dest_bugs:
      relative = dest_bugs[-1]
    else:
      for dir in Direction.flat():
        neighbor_bugs = self.bugs_from_pos(self.get_neighbor(move.destination, dir))
        if neighbor_bugs:
          relative = neighbor_bugs[0]
          direction = dir
          break
    return Move.stringify(moved, relative, direction)
  
  # def parse_move(self, move: str) -> Move:

  def bugs_from_pos(self, position: Position) -> list[Bug]:
    """
    Retrieves the list of bug pieces from the given position.

    :param position: Tile position.
    :type position: Position
    :return: The list of bug pieces at the given position.
    :rtype: list[Bug]
    """
    return self._pos_to_bug[position] if position in self._pos_to_bug else []

  def pos_from_bug(self, bug: Bug) -> Position | None:
    """
    Retrieves the position of the given bug piece.

    :param bug: Bug piece to get the position of.
    :type bug: Bug
    :return: Position of the given bug piece.
    :rtype: Position | None
    """
    return self._bug_to_pos[bug] if bug in self._bug_to_pos else None

  def get_valid_moves(self) -> Set[Move]:
    if not self._valid_moves_cache:
      self._valid_moves_cache = set()
      if self.state is GameState.NOT_STARTED or self.state is GameState.IN_PROGRESS:
        for bug, pos in self._bug_to_pos.items():
          # Iterate over available pieces of the current player
          if bug.color is self.current_player_color:
            # Turn 0 is White player's first turn
            if self.turn == 0:
              # Can't place the queen on the first turn
              if bug.type is not BugType.QUEEN_BEE and self.can_bug_be_played(bug):
                # Add the only valid placement for the current bug piece
                self._valid_moves_cache.add(Move(bug, None, self.ORIGIN))
            # Turn 0 is Black player's first turn
            elif self.turn == 1:
              # Can't place the queen on the first turn
              if bug.type is not BugType.QUEEN_BEE and self.can_bug_be_played(bug):
                # Add all valid placements for the current bug piece (can be placed only around the first White player's first piece)
                self._valid_moves_cache.update([Move(bug, None, self.get_neighbor(self.ORIGIN, direction)) for direction in Direction.flat()])
            # Bug piece has not been played yet
            elif not pos:
              # Check hand placement, and turn and queen placement, related rule.
              if self.can_bug_be_played(bug) and (self.current_player_turn != 4 or (self.current_player_turn == 4 and (self.current_player_queen_in_play or (not self.current_player_queen_in_play and bug.type is BugType.QUEEN_BEE)))):
                # Add all valid placements for the current bug piece
                self._valid_moves_cache.update([Move(bug, None, placement) for placement in self.get_valid_placements()])
            # A bug piece in play can move only if it's at the top and its queen is in play 
            elif self.current_player_queen_in_play and self.bugs_from_pos(pos)[-1] == bug:
              # Can't move pieces that would break the hive. Pieces stacked upon other can never break the hive by moving.
              if self.bugs_from_pos(pos) or self.can_move_without_breaking_hive(pos):
                pass
              else:
                pass
    return self._valid_moves_cache
  
  def get_best_move(self) -> Move | None:
    """
    Get the best move to play.  
    Currently, simply returns the first valid move.

    :return: The best possible move to play.
    :rtype: Move
    """
    return list(self.get_valid_moves())[0] if self.get_valid_moves() else None

  def can_bug_be_played(self, piece: Bug) -> bool:
    """
    Checks whether the given bug piece can be drawn from hand (has the lowest ID among same-type pieces).

    :param piece: Bug piece to check.
    :type piece: Bug
    :return: Whether the given bug piece can be played.
    :rtype: bool
    """
    return all(bug.id >= piece.id for bug, pos in self._bug_to_pos.items() if pos is None and bug.type is piece.type)

  def is_bug_on_top(self, bug: Bug) -> bool:
    """
    Checks if the given bug has been played and is at the top of the stack.

    :param bug: Bug piece.
    :type bug: Bug
    :return: Whether the bug is at the top.
    :rtype: bool
    """
    pos = self.pos_from_bug(bug)
    return pos != None and self.bugs_from_pos(pos)[-1] == bug

  def get_valid_placements(self) -> Set[Position]:
    """
    Calculates all valid placements for the current player.

    :return: Set of valid positions where new pieces can be placed.
    :rtype: Set[Position]
    """
    placements: Set[Position] = set()
    # Iterate over all placed bug pieces of the current player
    for bug, pos in self._bug_to_pos.items():
      if bug.color is self.current_player_color and pos and self.is_bug_on_top(bug):
        # Iterate over all neighbors of the current bug piece
        for direction in Direction.flat():
          neighbor = self.get_neighbor(pos, direction)
          # If the neighboring tile is empty
          if not self.bugs_from_pos(neighbor):
            # Iterate over all neighbor's neighbors, excluding the current bug piece
            for dir in Direction.flat():
              if dir is not direction.opposite:
                bugs = self.bugs_from_pos(self.get_neighbor(neighbor, dir))
                # Add the bug piece neighbor only if it does not border with an opponent's piece
                if not bugs or bugs[-1].color is self.current_player_color:
                  placements.add(neighbor)
    return placements

  def get_neighbor(self, position: Position, direction: Direction) -> Position:
    return position + self.NEIGHBOR_DELTAS[direction.delta_index]

  def can_move_without_breaking_hive(self, position: Position) -> bool:
    # Try gaps heuristic first
    neighbors: list[list[Bug]] = [self.bugs_from_pos(self.get_neighbor(position, direction)) for direction in Direction.flat()]
    gaps: int = sum(bool(neighbors[i] and not neighbors[i - 1]) for i in range(len(neighbors)))
    # If there is more than 1 gap, perform a DFS to check if all neighbors are still connected in some way.
    if gaps > 1:
      visited: Set[Position] = set()
      neighbors_pos: list[Position] = [pos for bugs in neighbors for pos in [self.pos_from_bug(bugs[-1])] if bugs if pos]    
      stack = [neighbors_pos[0]]
      while stack:
        current = stack.pop()
        visited.add(current)
        # Explore neighbors of the current position
        for direction in Direction.flat():
          neighbor = self.get_neighbor(current, direction)
          if neighbor != position and self.bugs_from_pos(neighbor) and neighbor not in visited:
            stack.append(neighbor)
      # Check if all neighbors with bug pieces were visited
      return all(neighbor_pos in visited for neighbor_pos in neighbors_pos)
    # If there is only 1 gap, then all neighboring pieces are connected even without the piece at the given position.
    return True

  def get_queen_bee_moves(self, bug: Bug, origin: Position):
    pass

  def get_spider_moves(self, bug: Bug, origin: Position):
    pass

  def get_beetle_moves(self, bug: Bug, origin: Position) -> Set[Move]:
    """
    Calculates the set of valid moves for a Beetle.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid Beetle moves.
    :rtype: Set[Move]
    """
    moves: Set[Move] = set()
    for direction in Direction.flat():
      height = len(self.bugs_from_pos(origin)) - 1 # Don't consider the Beetle
      destination = self.get_neighbor(origin, direction)
      dest_height = len(self.bugs_from_pos(destination))
      left_height = len(self.bugs_from_pos(self.get_neighbor(origin, direction.left_of)))
      right_height = len(self.bugs_from_pos(self.get_neighbor(origin, direction.right_of)))
      # Logic from http://boardgamegeek.com/wiki/page/Hive_FAQ#toc9
      if not ((height == 0 and dest_height == 0 and left_height == 0 and right_height == 0) or (dest_height < left_height and dest_height < right_height and height < left_height and height < right_height)):
        moves.add(Move(bug, origin, destination)) # TODO: Is the height / direction needed?
    return moves

  def get_grasshopper_moves(self, bug: Bug, origin: Position) -> Set[Move]:
    """
    Calculates the set of valid moves for a Grasshopper.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid Grasshopper moves.
    :rtype: Set[Move]
    """
    moves: Set[Move] = set()
    for direction in Direction.flat():
      destination: Position = self.get_neighbor(origin, direction)
      distance: int = 0
      while self.bugs_from_pos(destination):
        # Jump one more tile in the same direction
        destination = self.get_neighbor(origin, direction)
        distance += 1
      if distance > 0:
        # Can only move if there's at least one piece in the way
        moves.add(Move(bug, origin, destination))
    return moves

  def get_soldier_ant_moves(self, bug: Bug, origin: Position):
    pass

  def get_mosquito_moves(self, bug: Bug, origin: Position):
    pass

  def get_ladybug_moves(self, bug: Bug, origin: Position) -> Set[Move]:
    """
    Calculates the set of valid moves for a Ladybug.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid Ladybug moves.
    :rtype: Set[Move]
    """
    return {
      Move(bug, origin, origin)
      for first_move in self.get_beetle_moves(bug, origin) if len(self.bugs_from_pos(first_move.destination))
      for second_move in self.get_beetle_moves(bug, first_move.destination) if len(self.bugs_from_pos(second_move.destination))
      for final_move in self.get_beetle_moves(bug, second_move.destination) if len(self.bugs_from_pos(final_move.destination)) and final_move.destination != origin
    }

  def get_pillbug_moves(self, bug: Bug, origin: Position):
    pass

  def get_valid_slides(self, bug: Bug, origin: Position):
    pass
