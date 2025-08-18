from typing import Final, Optional, Set
from enums import GameType, GameState, PlayerColor, BugType, Direction
from game import Position, Bug, Move
import re
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional
from copy import deepcopy
import dictionaries
from collections.abc import Iterable

ACTION_SPACE_SHAPE = (28, 7, 14)
ACTION_SPACE_SIZE = np.prod(ACTION_SPACE_SHAPE)
MAX_TURNS = 100

class Board():
  """
  Game Board.
  """
  ORIGIN: Final[Position] = Position(0, 0)
  """
  Position of the first piece played.
  """
  NEIGHBOR_DELTAS: Final[tuple[Position, Position, Position, Position, Position, Position, Position, Position]] = (
    Position(1, 0), # Right
    Position(1, -1), # Up right
    Position(0, -1), # Up left
    Position(-1, 0), # Left
    Position(-1, 1), # Down left
    Position(0, 1), # Down right
    Position(0, 0), # Below (no change)
    Position(0, 0), # Above (no change)
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
    expansions = self.type if isinstance(self.type, Iterable) else [self.type] # type: ignore
    self.state: GameState = state
    self.turn: int = turn
    self.move_strings: list[str] = []
    self.moves: list[Optional[Move]] = []
    self._valid_moves_cache: Optional[Set[Move]] = None
    self._pos_to_bug: dict[Position, list[Bug]] = {}
    """
    Map for tile positions on the board and bug pieces placed there (pieces can be stacked).
    """
    self._bug_to_pos: dict[Bug, Optional[Position]] = {}
    """
    Numer of times that the same board state has been seen. Used for draw detection.
    """
    self.board_states: dict[str, int] = {}
    """
    Map for bug pieces and their current position.  
    Position is None if the piece has not been played yet.  
    Also serves as a check for valid pieces for the game.
    """
    for color in PlayerColor:
      for expansion in expansions:
        if expansion is GameType.Base:
          self._bug_to_pos[Bug(color, BugType.QUEEN_BEE)] = None
          # Add ids greater than 0 only for bugs with multiple copies.
          for i in range(1, 3):
            self._bug_to_pos[Bug(color, BugType.SPIDER, i)] = None
            self._bug_to_pos[Bug(color, BugType.BEETLE, i)] = None
            self._bug_to_pos[Bug(color, BugType.GRASSHOPPER, i)] = None
            self._bug_to_pos[Bug(color, BugType.SOLDIER_ANT, i)] = None
          self._bug_to_pos[Bug(color, BugType.GRASSHOPPER, 3)] = None
          self._bug_to_pos[Bug(color, BugType.SOLDIER_ANT, 3)] = None
        else:
          self._bug_to_pos[Bug(color, BugType(expansion.name))] = None
    self._play_initial_moves(moves)

  def __str__(self) -> str:
    return f"{self.type};{self.state};{self.current_player_color}[{self.current_player_turn}]{';' if len(self.moves) else ''}{';'.join(self.move_strings)}"

  @property
  def current_player_color(self) -> PlayerColor:
    """
    Color of the current player.

    :rtype: PlayerColor
    """
    return PlayerColor.WHITE if self.turn % 2 == 0 else PlayerColor.BLACK

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
    return bool(self._bug_to_pos[Bug(self.current_player_color, BugType.QUEEN_BEE)])

  @property
  def valid_moves(self) -> str:
    """
    Current possible legal moves in a joined list of MoveStrings.

    :rtype: str
    """
    return ";".join([self.stringify_move(move) for move in self._get_valid_moves()]) or Move.PASS
  
  def valid_moves_complete(self) -> str:
    """
    Current possible legal moves in a joined list of MoveStrings, including all combinations.

    :rtype: str
    """
    return ";".join([self.stringify_move(move, True) for move in self._get_valid_moves()]) or Move.PASS

  def play(self, move_string: str) -> None:
    """
    Plays the given move.

    :param move_string: MoveString of the move to play.
    :type move_string: str
    :raises ValueError: If the game is over.
    """
    move = self._parse_move(move_string)
    if self.state is GameState.NOT_STARTED:
      self.state = GameState.IN_PROGRESS
    if self.state is GameState.IN_PROGRESS:
      self.turn += 1
      self.move_strings.append(move_string)
      self.moves.append(move)
      self._valid_moves_cache = None
      if move:
        self._bug_to_pos[move.bug] = move.destination
        if move.origin:
          self._pos_to_bug[move.origin].pop()
        if move.destination in self._pos_to_bug:
          self._pos_to_bug[move.destination].append(move.bug)
        else:
          self._pos_to_bug[move.destination] = [move.bug]
        actual_encoding = str(self.current_player_color.code) + self.stringRepresentation()
        if actual_encoding in self.board_states:
          self.board_states[actual_encoding] += 1
        else:
          self.board_states[actual_encoding] = 1
        black_queen_surrounded = (queen_pos := self._bug_to_pos[Bug(PlayerColor.BLACK, BugType.QUEEN_BEE)]) and all(self._bugs_from_pos(self._get_neighbor(queen_pos, direction)) for direction in Direction.flat())
        white_queen_surrounded = (queen_pos := self._bug_to_pos[Bug(PlayerColor.WHITE, BugType.QUEEN_BEE)]) and all(self._bugs_from_pos(self._get_neighbor(queen_pos, direction)) for direction in Direction.flat())
        if black_queen_surrounded and white_queen_surrounded or self.board_states[actual_encoding] >= 3 or self.turn >= MAX_TURNS:
          self.state = GameState.DRAW
        elif black_queen_surrounded:
          self.state = GameState.WHITE_WINS
        elif white_queen_surrounded:
          self.state = GameState.BLACK_WINS
    else:
      raise ValueError(f"You can't {'play' if move else Move.PASS} when the game is over")

  def undo(self, amount: int = 1) -> None:
    """
    Undoes the specified amount of moves.

    :param amount: Amount of moves to undo, defaults to 1.
    :type amount: int, optional
    :raises ValueError: If there are not enough moves to undo.
    :raises ValueError: If the game has yet to begin.
    """
    if self.state is not GameState.NOT_STARTED:
      if len(self.moves) >= amount:
        if self.state is not GameState.IN_PROGRESS:
          self.state = GameState.IN_PROGRESS
        for _ in range(amount):
          self.turn -= 1
          self.move_strings.pop()
          move = self.moves.pop()
          self._valid_moves_cache = None
          if move:
            self._pos_to_bug[move.destination].pop()
            self._bug_to_pos[move.bug] = move.origin
            if move.origin:
              self._pos_to_bug[move.origin].append(move.bug)
        if self.turn == 0:
          self.state = GameState.NOT_STARTED
      else:
        raise ValueError(f"Not enough moves to undo: asked for {amount} but only {len(self.moves)} were made")
    else:
      raise ValueError(f"The game has yet to begin")

  def stringify_move(self, move: Optional[Move], all_combinations: bool = False) -> str:
    """
    Returns a MoveString from the given move.

    :param move: Move.
    :type move: Optional[Move]
    :return: MoveString.
    :rtype: str
    """
    if move:
      moved: Bug = move.bug
      relative: Optional[Bug] = None
      direction: Optional[Direction] = None
      results: list[str] = []
      if (dest_bugs := self._bugs_from_pos(move.destination)):
        relative = dest_bugs[-1]
        results.append(Move.stringify(moved, relative, direction))
      else:
        for dir in Direction.flat():
            for neighbor_bug in self._bugs_from_pos(self._get_neighbor(move.destination, dir)):
              if neighbor_bug != moved:
                relative = neighbor_bug
                direction = dir.opposite
                results.append(Move.stringify(moved, relative, direction))
                if not all_combinations:
                  break
              if not all_combinations:
                break
      if not results:
        results.append(Move.stringify(moved, None, None))
      return ";".join(results)
    return Move.PASS

  def encode_board(self, grid_size: int = 26) -> NDArray[np.int64]:
      """
      Encodes the current board state into a 4-plane numpy array.
      
      Planes:
        - Plane 0: Binary plane for White queen (1 if present, 0 otherwise).
        - Plane 1: Binary plane for Black queen.
        - Plane 2: White non-queen pieces with integer values (1 to 7).
        - Plane 3: Black non-queen pieces with integer values (1 to 7).
      
      The board's origin (0,0) is mapped to the center of the grid.
      
      :param board: The current game board.
      :param grid_size: The size of the grid along one dimension.
      :return: A numpy array of shape (4, grid_size, grid_size) representing the board.
      """
      N = grid_size
      offset = N // 2

      encoding = np.zeros((4, N, N), dtype=np.int64)
      
      nonqueen_mapping = {
      "A": 1,  # Soldier Ant
      "G": 2,  # Grasshopper
      "B": 3,  # Beetle
      "S": 4,  # Spider
      "M": 5,  # Mosquito
      "L": 6,  # Ladybug
      "P": 7   # Pillbug
      }

      for pos, bugs in self._pos_to_bug.items():
        if not bugs:
            continue
        i = pos.q + offset
        j = pos.r + offset
        if i < 0 or i >= N or j < 0 or j >= N:
            continue  # Skip positions outside our grid.
        
        top_bug = bugs[-1]
        if top_bug.type == BugType.QUEEN_BEE:
            if top_bug.color == PlayerColor.WHITE:
                encoding[0, i, j] = 1
            else:
                encoding[1, i, j] = 1
        else:
            bug_code = str(top_bug.type)
            value = nonqueen_mapping.get(bug_code, 0)
            if top_bug.color == PlayerColor.WHITE:
                encoding[2, i, j] = value
            else:
                encoding[3, i, j] = value
                
      return encoding
  
  def stringRepresentation(self) -> str:
        """
        Returns a hashable string representation that uniquely identifies the current state of the board.
        
        This is done by encoding the board into a numpy array (using board.encode_board),
        flattening that array, and joining its elements into a string separated by colons.
        
        Adjust the grid_size as needed.
        """
        encoding = self.encode_board(grid_size=26)
        #print("White queen plane:\n", encoding[0])
        #print("Black queen plane:\n", encoding[1])
        #print("White non-queen pieces plane:\n", encoding[2])
        #print("Black non-queen pieces plane:\n", encoding[3])

        board_as_string = ":".join(encoding.astype(str).flatten().tolist())
        return board_as_string
  
  def encode_move_string(self, move_str: str, simple: bool = False) -> int:
    """
    Converts a move string (using BoardSpace notation) into an action index
    in the tile-relative action space.
    
    The action space is defined as (28, 7, 14) in full mode, with one extra index
    reserved for "pass". In non-simple mode, the index is computed as:
      index = (TILE_DIM * NUM_DIRECTIONS * next_to_encoding)
            + (TILE_DIM * direction_encoding)
            + tile_encoding
    where:
      - TILE_DIM = 14 (number of distinct tile codes)
      - NUM_DIRECTIONS = 7 (0–5 from explicit symbols plus 6 for stacking)
    
    Args:
        move_str: The move string, e.g.:
                  - "wB1 wS1" for a stacking move (beetle moves on top of spider)
                  - "bS1 wS1/" for a move with an explicit suffix direction indicator
                  - "wS1" for the first move.
        player: The player making the move (0 for dark, 1 for light).
        simple: If True, use simplified encoding (ignoring direction).
    
    Returns:
        An integer index in the range [0, ACTION_SPACE_SIZE] where ACTION_SPACE_SIZE = 28*7*14.
    """
    # If the move is "pass", return the reserved index.
    if move_str.strip().lower() == "pass":
        return int(28 * 7 * 14)
    
    parts = move_str.split()
    
    if len(parts) == 1:
        # First move case: only a piece's short name is given.
        moving_piece = parts[0]  # e.g., "wS1"
        tile_encoding = dictionaries.TILE_DICT_CANONICAL.get(moving_piece, 0)
        next_to_encoding = 0  # default value
        direction_encoding = 0  # default direction
    elif len(parts) == 2:
        moving_piece = parts[0]      # e.g., "wB1"
        destination = parts[1]         # e.g., "wS1/" or "wS1"
        tile_encoding = dictionaries.TILE_DICT_CANONICAL.get(moving_piece, 0)
        # Check if the destination contains an explicit direction indicator.
        if destination[0] in dictionaries.DIRECTION_MAP_PREFIX:
            direction_encoding = dictionaries.DIRECTION_MAP_PREFIX[destination[0]]
            neighbor_str = destination[1:]
        elif destination[-1] in dictionaries.DIRECTION_MAP_SUFFIX:
            direction_encoding = dictionaries.DIRECTION_MAP_SUFFIX[destination[-1]]
            neighbor_str = destination[:-1]
        else:
            # No explicit indicator => stacking move.
            direction_encoding = 6
            neighbor_str = destination
        # Use the full dictionary to get next_to_encoding.
        player = 1 if self.current_player_color == PlayerColor.WHITE else 0
        next_to_encoding = dictionaries.TILE_DICT_FULL[player].get(neighbor_str, 0)
    else:
        raise ValueError(f"Cannot parse move string: {move_str}")
    
    if simple:
        index = (14 * next_to_encoding) + tile_encoding
    else:
        index = (14 * 7 * next_to_encoding) + (14 * direction_encoding) + tile_encoding
    return int(index)
  
  def decode_move_index(self, index: int, simple: bool = False) -> str:
    # If index equals the reserved pass index:
    if index == ACTION_SPACE_SIZE:
        return "pass"

    # First: try to resolve the move by comparing against valid moves (and their synonyms).
    # This preserves the exact moved bug/color (e.g., pillbug moving opponent piece).
    for move in self._get_valid_moves():
        # primary stringification
        try:
            move_str = self.stringify_move(move)
            if self.encode_move_string(move_str, simple=simple) == index:
                return move_str
        except ValueError:
            pass

        # also try all synonym variants if not simple
        if not simple:
            try:
                all_synonyms = self.stringify_move(move, all_combinations=True).split(";")
                for synonym in all_synonyms:
                    if not synonym:
                        continue
                    try:
                        if self.encode_move_string(synonym, simple=simple) == index:
                            return synonym
                    except ValueError:
                        continue
            except ValueError:
                pass
  
  def invert_colors(self) -> 'Board':
    new = deepcopy(self)
    # flip every bug
    new._bug_to_pos = { b.invert_color():p for b,p in self._bug_to_pos.items() }
    new._pos_to_bug = { p:[b.invert_color() for b in bs] for p,bs in self._pos_to_bug.items() }
    new._valid_moves_cache = None
    return new
  
  @staticmethod
  def _rotate60_pos(pos: Position) -> Position:
    # using cube-coords rotation (x,y,z)->(−z,−x,−y),
    # with x=q, y=r, z=−q−r  ⇒  axial (q,r)->(q+r, −q)
    return Position(pos.q + pos.r, -pos.q)

  @staticmethod
  def _reflect_vertically_pos(pos: Position) -> Position:
    # flip rows (i.e. invert q)
    return Position(-pos.q, pos.r)
  
  from typing import Dict, Optional

  def rotate_60(self) -> "Board":
      """Return a new Board rotated +60° around the origin."""
      new = deepcopy(self)

      new_map: Dict[Bug, Optional[Position]] = {bug: None for bug in self._bug_to_pos}
      for bug, pos in self._bug_to_pos.items():
          if pos is not None:
              new_map[bug] = Board._rotate60_pos(pos)
      new._bug_to_pos = new_map

      new._pos_to_bug = {}
      for bug, pos in new_map.items():
          if pos is not None:
              new._pos_to_bug.setdefault(pos, []).append(bug)

      new._valid_moves_cache = None

      return new


  def reflect_vertically(self) -> "Board":
      """Return a new Board flipped top↔bottom across the horizontal axis."""
      new = deepcopy(self)

      new_map: Dict[Bug, Optional[Position]] = {bug: None for bug in self._bug_to_pos}
      for bug, pos in self._bug_to_pos.items():
          if pos is not None:
              new_map[bug] = Board._reflect_vertically_pos(pos)
      new._bug_to_pos = new_map

      new._pos_to_bug = {}
      for bug, pos in new_map.items():
          if pos is not None:
              new._pos_to_bug.setdefault(pos, []).append(bug)

      new._valid_moves_cache = None
      return new

  def _parse_turn(self, turn: str) -> int:
    """
    Parses a TurnString.

    :param turn: TurnString.
    :type turn: str
    :raises ValueError: If it's not a valid TurnString.
    :return: Turn number.
    :rtype: int
    """
    if (match := re.fullmatch(f"({PlayerColor.WHITE}|{PlayerColor.BLACK})\\[(\\d+)\\]", turn)):
      color, player_turn = match.groups()
      if (turn_number := int(player_turn)) > 0:
        return 2 * turn_number - 2 + list(PlayerColor).index(PlayerColor(color))
      else:
        raise ValueError(f"The turn number must be greater than 0")
    else:
      raise ValueError(f"'{turn}' is not a valid TurnString")

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

  def _play_initial_moves(self, moves: list[str]) -> None:
    """
    Make initial moves.

    :param moves: List of MoveStrings.
    :type moves: list[str]
    :raises ValueError: If the amount of moves to make is not coherent with the turn number.
    """
    if self.turn == len(moves):
      old_turn = self.turn
      old_state = self.state
      self.turn = 0
      self.state = GameState.NOT_STARTED
      for move in moves:
        self.play(move)
      if old_turn != self.turn:
        raise ValueError(f"TurnString is not correct, should be {self.current_player_color}[{self.current_player_turn}]")
      if old_state != self.state:
        raise ValueError(f"GameStateString is not correct, should be {self.state}")
    else:
      raise ValueError(f"Expected {self.turn} moves but got {len(moves)}")

  def _get_valid_moves(self) -> Set[Move]:
    """
    Calculates the set of valid moves for the current player.

    :return: Set of valid moves.
    :rtype: Set[Move]
    """
    if not self._valid_moves_cache:
      self._valid_moves_cache = set()
      if self.state is GameState.NOT_STARTED or self.state is GameState.IN_PROGRESS:
        for bug, pos in self._bug_to_pos.items():
          # Iterate over available pieces of the current player
          if bug.color is self.current_player_color:
            # Turn 0 is White player's first turn
            if self.turn == 0:
              # Can't place the queen on the first turn
              if bug.type is not BugType.QUEEN_BEE and self._can_bug_be_played(bug):
                # Add the only valid placement for the current bug piece
                self._valid_moves_cache.add(Move(bug, None, self.ORIGIN))
            # Turn 0 is Black player's first turn
            elif self.turn == 1:
              # Can't place the queen on the first turn
              if bug.type is not BugType.QUEEN_BEE and self._can_bug_be_played(bug):
                # Add all valid placements for the current bug piece (can be placed only around the first White player's first piece)
                self._valid_moves_cache.update(Move(bug, None, self._get_neighbor(self.ORIGIN, direction)) for direction in Direction.flat())
            # Bug piece has not been played yet
            elif not pos:
              # Check hand placement, and turn and queen placement, related rule.
              if self._can_bug_be_played(bug) and (self.current_player_turn != 4 or (self.current_player_turn == 4 and (self.current_player_queen_in_play or (not self.current_player_queen_in_play and bug.type is BugType.QUEEN_BEE)))):
                # Add all valid placements for the current bug piece
                self._valid_moves_cache.update(Move(bug, None, placement) for placement in self._get_valid_placements())
            # A bug piece in play can move only if it's at the top and its queen is in play and has not been moved in the previous player's turn
            elif self.current_player_queen_in_play and self._bugs_from_pos(pos)[-1] == bug and self._was_not_last_moved(bug):
              # Can't move pieces that would break the hive. Pieces stacked upon other can never break the hive by moving
              if len(self._bugs_from_pos(pos)) > 1 or self._can_move_without_breaking_hive(pos):
                match bug.type:
                  case BugType.QUEEN_BEE:
                    self._valid_moves_cache.update(self._get_sliding_moves(bug, pos, 1))
                  case BugType.SPIDER:
                    self._valid_moves_cache.update(self._get_spider_moves(bug, pos))
                  case BugType.BEETLE:
                    self._valid_moves_cache.update(self._get_beetle_moves(bug, pos))
                  case BugType.GRASSHOPPER:
                    self._valid_moves_cache.update(self._get_grasshopper_moves(bug, pos))
                  case BugType.SOLDIER_ANT:
                    self._valid_moves_cache.update(self._get_sliding_moves(bug, pos))
                  case BugType.MOSQUITO:
                    self._valid_moves_cache.update(self._get_mosquito_moves(bug, pos))
                  case BugType.LADYBUG:
                    self._valid_moves_cache.update(self._get_ladybug_moves(bug, pos))
                  case BugType.PILLBUG:
                    self._valid_moves_cache.update(self._get_sliding_moves(bug, pos, 1))
                    self._valid_moves_cache.update(self._get_pillbug_special_moves(pos))
              else:
                match bug.type:
                  case BugType.MOSQUITO:
                    self._valid_moves_cache.update(self._get_mosquito_moves(bug, pos, True))
                  case BugType.PILLBUG:
                    self._valid_moves_cache.update(self._get_pillbug_special_moves(pos))
                  case _:
                    pass
    return self._valid_moves_cache

  def _get_valid_placements(self) -> Set[Position]:
    """
    Calculates all valid placements for the current player.

    :return: Set of valid positions where new pieces can be placed.
    :rtype: Set[Position]
    """
    placements: Set[Position] = set()
    # Iterate over all placed bug pieces of the current player
    for bug, pos in self._bug_to_pos.items():
      if bug.color is self.current_player_color and pos and self._is_bug_on_top(bug):
        # Iterate over all neighbors of the current bug piece
        for direction in Direction.flat():
          neighbor = self._get_neighbor(pos, direction)
          # If the neighboring tile is empty
          if not self._bugs_from_pos(neighbor):
            # If all neighbor's neighbors are empty or of the same color, add the neighbor as a valid placement
            if all(not self._bugs_from_pos(self._get_neighbor(neighbor, dir)) or self._bugs_from_pos(self._get_neighbor(neighbor, dir))[-1].color is self.current_player_color for dir in Direction.flat() if dir is not direction.opposite):
              placements.add(neighbor)
    return placements

  def _get_sliding_moves(self, bug: Bug, origin: Position, depth: int = 0) -> Set[Move]:
    """
    Calculates the set of valid sliding moves, optionally with a fixed depth.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :param depth: Optional fixed depth of the move, defaults to `0`.
    :type depth: int, optional
    :return: Set of valid sliding moves.
    :rtype: Set[Move]
    """
    destinations: Set[Position] = set()
    visited: Set[Position] = set()
    queue: Set[tuple[Position, int]] = {(origin, 0)}
    unlimited_depth = depth == 0
    while queue:
      current, current_depth = queue.pop()
      visited.add(current)
      if unlimited_depth or current_depth == depth:
        destinations.add(current)
      if unlimited_depth or current_depth < depth:
        queue.update(
          (neighbor, current_depth + 1)
          for direction in Direction.flat()
          if (neighbor := self._get_neighbor(current, direction)) not in visited and not self._bugs_from_pos(neighbor) and self._check_for_door(origin, current, direction)
        )
    return {Move(bug, origin, destination) for destination in destinations if destination != origin}
  
  def _check_for_door(self, origin: Position, position: Position, direction: Direction) -> bool:
    """
    Checks whether a bug piece can slide from origin to position (no door formation).

    :param origin: Initial position of the bug piece.
    :type origin: Position
    :param position: Destination position.
    :type position: Position
    :param direction: Moving direction.
    :type direction: Direction
    :return: Whether a bug piece can slide from origin to position.
    :rtype: bool
    """
    right = self._get_neighbor(position, direction.clockwise)
    bug_on_right = False
    if right != origin:
      bug_on_right = bool(self._bugs_from_pos(right))
    left = self._get_neighbor(position, direction.anticlockwise)
    bug_on_left = False
    if left != origin:
      bug_on_left = bool(self._bugs_from_pos(left))
    return bug_on_right != bug_on_left

  def _get_spider_moves(self, bug: Bug, origin: Position) -> Set[Move]:
    """
    Calculates the set of valid spider moves.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid spider moves.
    :rtype: Set[Move]
    """
    destinations: Set[Position] = set()
    queue: Set[tuple[Position, int, Optional[Position]]] = {(origin, 0, None)}
    depth = 3
    while queue:
      current, current_depth, prev_pos = queue.pop()
      if current_depth == depth:
        destinations.add(current)
      if current_depth < depth:
        queue.update(
          (neighbor, current_depth + 1, current)
          for direction in Direction.flat()
          if (neighbor := self._get_neighbor(current, direction)) != prev_pos and not self._bugs_from_pos(neighbor) and self._check_for_door(origin, current, direction)
        )
    return {Move(bug, origin, destination) for destination in destinations if destination != origin}


  def _get_beetle_moves(self, bug: Bug, origin: Position, virtual: bool = False) -> Set[Move]:
    """
    Calculates the set of valid moves for a Beetle.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :param virtual: Whether the bug is not at origin, and is just passing by as part of its full move, defaults to False.
    :type virtual: bool, optional
    :return: Set of valid Beetle moves.
    :rtype: Set[Move]
    """
    moves: Set[Move] = set()
    for direction in Direction.flat():
      # Don't consider the Beetle in the height, unless it's a virtual move (the bug is not actually in origin, but moving at the top of origin is part of its full move).
      height = len(self._bugs_from_pos(origin)) - 1 + virtual
      destination = self._get_neighbor(origin, direction)
      dest_height = len(self._bugs_from_pos(destination))
      left_height = len(self._bugs_from_pos(self._get_neighbor(origin, direction.left_of)))
      right_height = len(self._bugs_from_pos(self._get_neighbor(origin, direction.right_of)))
      # Logic from http://boardgamegeek.com/wiki/page/Hive_FAQ#toc9
      if not ((height == 0 and dest_height == 0 and left_height == 0 and right_height == 0) or (dest_height < left_height and dest_height < right_height and height < left_height and height < right_height)):
        moves.add(Move(bug, origin, destination))
    return moves

  def _get_grasshopper_moves(self, bug: Bug, origin: Position) -> Set[Move]:
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
      destination: Position = self._get_neighbor(origin, direction)
      distance: int = 0
      while self._bugs_from_pos(destination):
        # Jump one more tile in the same direction
        destination = self._get_neighbor(destination, direction)
        distance += 1
      if distance > 0:
        # Can only move if there's at least one piece in the way
        moves.add(Move(bug, origin, destination))
    return moves

  def _get_mosquito_moves(self, bug: Bug, origin: Position, special_only: bool = False) -> Set[Move]:
    """
    Calculates the set of valid Mosquito moves, which copies neighboring bug pieces moves, and can be either normal or special (Pillbug) moves depending on the special_only flag.

    :param bug: Mosquito bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :param special_only: Whether to include special moves only, defaults to False.
    :type special_only: bool, optional
    :return: Set of valid Mosquito moves.
    :rtype: Set[Move]
    """
    # TODO: controllare HV-abstrato-WeakBot-2025-03-22-1655.pgn in cui un mosquito non riece a comportarsi come un pillbug
    if len(self._bugs_from_pos(origin)) > 1:
      return self._get_beetle_moves(bug, origin)
    moves: Set[Move] = set()
    bugs_copied: Set[BugType] = set()
    for direction in Direction.flat():
      if (bugs := self._bugs_from_pos(self._get_neighbor(origin, direction))) and (neighbor := bugs[-1]).type not in bugs_copied:
         bugs_copied.add(neighbor.type)
         if special_only:
           if neighbor.type == BugType.PILLBUG:
             moves.update(self._get_pillbug_special_moves(origin))
         else:
          match neighbor.type:
            case BugType.QUEEN_BEE:
              moves.update(self._get_sliding_moves(bug, origin, 1))
            case BugType.SPIDER:
              moves.update(self._get_spider_moves(bug, origin))
            case BugType.BEETLE:
              moves.update(self._get_beetle_moves(bug, origin))
            case BugType.GRASSHOPPER:
              moves.update(self._get_grasshopper_moves(bug, origin))
            case BugType.SOLDIER_ANT:
              moves.update(self._get_sliding_moves(bug, origin))
            case BugType.LADYBUG:
              moves.update(self._get_ladybug_moves(bug, origin))
            case BugType.PILLBUG:
              moves.update(self._get_sliding_moves(bug, origin, 1))
              moves.update(self._get_pillbug_special_moves(origin))
            case BugType.MOSQUITO:
              pass
    return moves

  def _get_ladybug_moves(self, bug: Bug, origin: Position) -> Set[Move]:
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
      Move(bug, origin, final_move.destination)
      for first_move in self._get_beetle_moves(bug, origin, True) if self._bugs_from_pos(first_move.destination)
      for second_move in self._get_beetle_moves(bug, first_move.destination, True) if self._bugs_from_pos(second_move.destination) and second_move.destination != origin
      for final_move in self._get_beetle_moves(bug, second_move.destination, True) if not self._bugs_from_pos(final_move.destination) and final_move.destination != origin
    }

  def _get_pillbug_special_moves(self, origin: Position) -> Set[Move]:
      """
      Calculates the set of valid special Pillbug moves:
      - Pick up an adjacent single (non-stacked) bug (any color),
        provided it wasn't last moved and lifting it doesn't break the hive,
        and place it on another empty neighboring tile of the pillbug.
      """
      moves: Set[Move] = set()
      # Candidate drop targets: empty neighbors of the pillbug
      empty_positions = [self._get_neighbor(origin, d) for d in Direction.flat() if not self._bugs_from_pos(self._get_neighbor(origin, d))]
      if not empty_positions:
          return moves  # nothing to place to

      for direction in Direction.flat():
          pos = self._get_neighbor(origin, direction)
          bugs = self._bugs_from_pos(pos)
          if len(bugs) != 1:
              continue  # must be a single bug (not stacked)
          neighbor = bugs[-1]
          if not self._was_not_last_moved(neighbor):
              continue
          if not self._can_move_without_breaking_hive(pos):
              continue
          # For each empty placement adjacent to the pillbug
          for dest in empty_positions:
              if dest == pos:
                  continue  # can't "move" to the same square
              moves.add(Move(neighbor, pos, dest))
      return moves

  def _can_move_without_breaking_hive(self, position: Position) -> bool:
    """
    Checks whether a bug piece can be moved from the given position.

    :param position: Position where the bug piece is located.
    :type position: Position
    :return: Whether a bug piece in the given position can move.
    :rtype: bool
    """
    # Try gaps heuristic first
    neighbors: list[list[Bug]] = [self._bugs_from_pos(self._get_neighbor(position, direction)) for direction in Direction.flat()]
    # If there is more than 1 gap, perform a DFS to check if all neighbors are still connected in some way.
    if sum(bool(neighbors[i] and not neighbors[i - 1]) for i in range(len(neighbors))) > 1:
      visited: Set[Position] = set()
      neighbors_pos: list[Position] = [pos for bugs in neighbors if bugs and (pos := self._pos_from_bug(bugs[-1]))]    
      stack: Set[Position] = {neighbors_pos[0]}
      while stack:
        current = stack.pop()
        visited.add(current)
        stack.update(neighbor for direction in Direction.flat() if (neighbor := self._get_neighbor(current, direction)) != position and self._bugs_from_pos(neighbor) and neighbor not in visited)
      # Check if all neighbors with bug pieces were visited
      return all(neighbor_pos in visited for neighbor_pos in neighbors_pos)
    # If there is only 1 gap, then all neighboring pieces are connected even without the piece at the given position.
    return True

  def _can_bug_be_played(self, piece: Bug) -> bool:
    """
    Checks whether the given bug piece can be drawn from hand (has the lowest ID among same-type pieces).

    :param piece: Bug piece to check.
    :type piece: Bug
    :return: Whether the given bug piece can be played.
    :rtype: bool
    """
    return all(bug.id >= piece.id for bug, pos in self._bug_to_pos.items() if pos is None and bug.type is piece.type and bug.color is piece.color)

  def _was_not_last_moved(self, bug: Bug) -> bool:
    """
    Checks whether the given bug piece was not moved in the previous turn.

    :param bug: Bug piece.
    :type bug: Bug
    :return: Whether the bug piece was not last moved.
    :rtype: bool
    """
    return not self.moves[-1] or self.moves[-1].bug != bug

  def _parse_move(self, move_string: str) -> Optional[Move]:
    """
    Parses a MoveString.

    :param move_string: MoveString.
    :type move_string: str
    :raises ValueError: If move_string is 'pass' but there are other valid moves.
    :raises ValueError: If move_string is not a valid move for the current board state.
    :raises ValueError: If bug_string_2 has not been played yet.
    :raises ValueError: If more than one direction was specified.
    :raises ValueError: If move_string is not a valid MoveString.
    :return: Move.
    :rtype: Optional[Move]
    wL wG1/;wL wG1-;wA1 wP/;wA1 \\bS2;wL wA1-;wA1 \\bQ;wA1 -bQ;wA1 wG1\\;wG2 /wS1;wL /wG1;wL bS2\\;wG2 bS2/;wG2 \\wG1;wB2 bS2/;wB2 \\wG1;wA2 wP/;wA2 \\bS2;wA1 -bS1;wA1 /wS2;wA1 -wQ;wA1 /bS1;wA1 wM/;wA1 \\wB1;wL \\wA1;wA2 wG1\\;wA1 wS1\\;wA1 bB2\\;wA1 bS1-;wA1 wQ/;wA1 \\wP;wS1 /bS2;wS1 wP\\;wS1 bL-;wA1 \\wS2;wB2 wP/;wB2 \\bS2;wA2 wS1\\;wA2 \\wS2;wA2 wA1/;wA1 wG1/;wA1 -bM;wA1 /bQ;wA1 wG1-;wB2 wP;wA1 /wG1;wA1 bS2\\;wA2 wG1/;wA2 wG1-;wG2 wP/;wG2 \\bS2;wA2 wA1-;wA1 bQ/;wA1 bQ-;wA1 bM/;wA1 \\wM;wA1 /bM;wA1 -wS1;wA1 /wM;wA1 bM\\;wA2 /wG1;wA2 bS2\\;wG2 wG1\\;wA1 bL\\;wG2 wA1/;wG2 wS1\\;wS2 wA1/;wA2 \\wA1;bL /bS2;bL wP\\;wB2 /wG1;wB2 bS2\\;wG2 \\wS2;wL -wS2;wG2 wG1/;wG2 wG1-;wA1 /bL;wA1 wB1\\;wA1 wS1-;wG2 /wG1;wG2 bS2\\;wG2 wA1-;wS2 bQ-;wS2 bM/;wS2 \\wM;wL /wS1;wL bS2/;wL \\wG1;wA1 bB2-;wG2 \\wA1;wA1 /bS2;wA1 wP\\;wA1 bL-;wA1 -wS2;wS1 -bM;wS1 /bQ;wA2 -wS2;wA1 /wS1;wG1 -wQ;wG1 /bS1;wG1 wM/;wG1 \\wB1;wB2 /bS2;wB2 wP\\;wB2 bL-;wL wP/;wL \\bS2;wA1 bS2/;wA1 \\wG1;wL wG1\\;wA2 /wS1;wL wS1\\;wA2 bS2/;wA2 \\wG1;wA1 wS2/;wA1 \\bB2;wL wA1/;wG2 -wS2;wL \\wS2;wB2 wG1
    """
    if move_string == Move.PASS:
      return None
    if (match := re.fullmatch(Move.REGEX, move_string)):
      bug_string_1, _, _, _, _, left_dir, bug_string_2, _, _, _, right_dir = match.groups()
      if not left_dir or not right_dir:
        moved = Bug.parse(bug_string_1)
        if (relative_pos := self._pos_from_bug(Bug.parse(bug_string_2)) if bug_string_2 else self.ORIGIN):
          if left_dir:
              dir_str = f"{left_dir}|"
          else:
              dir_str = f"|{right_dir}" if right_dir else ""
          direction = Direction(dir_str)
          move = Move(moved, self._pos_from_bug(moved), self._get_neighbor(relative_pos, direction))
          return move
        raise ValueError(f"'{bug_string_2}' has not been played yet")
      raise ValueError(f"Only one direction at a time can be specified")
    raise ValueError(f"'{move_string}' is not a valid MoveString")

  def _is_bug_on_top(self, bug: Bug) -> bool:
    """
    Checks if the given bug has been played and is at the top of the stack.

    :param bug: Bug piece.
    :type bug: Bug
    :return: Whether the bug is at the top.
    :rtype: bool
    """
    return (pos := self._pos_from_bug(bug)) != None and self._bugs_from_pos(pos)[-1] == bug

  def _bugs_from_pos(self, position: Position) -> list[Bug]:
    """
    Retrieves the list of bug pieces from the given position.

    :param position: Tile position.
    :type position: Position
    :return: The list of bug pieces at the given position.
    :rtype: list[Bug]
    """
    return self._pos_to_bug[position] if position in self._pos_to_bug else []

  def _pos_from_bug(self, bug: Bug) -> Optional[Position]:
    """
    Retrieves the position of the given bug piece.

    :param bug: Bug piece to get the position of.
    :type bug: Bug
    :return: Position of the given bug piece.
    :rtype: Optional[Position]
    """
    return self._bug_to_pos[bug] if bug in self._bug_to_pos else None

  def _get_neighbor(self, position: Position, direction: Direction) -> Position:
    """
    Returns the neighboring position from the given direction.

    :param position: Central position.
    :type position: Position
    :param direction: Direction of movement.
    :type direction: Direction
    :return: Neighboring position in the specified direction.
    :rtype: Position
    """
    return position + self.NEIGHBOR_DELTAS[direction.delta_index]

  def get_bug_positions(self) -> Dict[Bug, Optional[Position]]:
    """Returns a copy of the bug-to-position mapping."""
    return self._bug_to_pos.copy()
    
      
