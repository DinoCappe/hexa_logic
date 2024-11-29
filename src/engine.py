from typing import TypeGuard, Final, Optional, Any
from copy import deepcopy
from enums import Command, Option, OptionType, Strategy, PlayerColor
from board import Board
from game import Move
from ai import Brain, Random, AlphaBetaPruner
import os
import re

class Engine():
  """
  Game engine.
  """

  VERSION: Final[str] = "1.2.0"
  """
  Engine version.
  """
  OPTION_TYPES: Final[dict[Option, OptionType]] = {
    Option.STRATEGY_WHITE: OptionType.ENUM,
    Option.STRATEGY_BLACK: OptionType.ENUM,
    Option.NUM_THREADS: OptionType.INT
  }
  """
  Map for options and their type.
  """

  BRAINS: Final[dict[Strategy, Brain]] = {
    Strategy.RANDOM: Random(),
    Strategy.MINMAX: AlphaBetaPruner()
  }
  """
  Map for strategies and the respective brain.
  """
  DEFAULT_STRATEGY_WHITE: Final[Strategy] = Strategy.MINMAX
  """
  Default value for option StrategyWhite.
  """
  DEFAULT_STRATEGY_BLACK: Final[Strategy] = Strategy.MINMAX
  """
  Default value for option StrategyBlack.
  """

  MIN_NUM_THREADS: Final[int] = 1
  """
  Minimum value for option NumThreads.
  """
  DEFAULT_NUM_THREADS: Final[int] = threads // 2 if (threads := os.cpu_count()) else MIN_NUM_THREADS
  """
  Default value for option NumThreads.
  """
  MAX_NUM_THREADS: Final[int] = threads if (threads := os.cpu_count()) else MIN_NUM_THREADS
  """
  Maximum value for option NumThreads.
  """

  def __init__(self) -> None:
    self.strategywhite: Strategy = self.DEFAULT_STRATEGY_WHITE
    self.strategyblack: Strategy = self.DEFAULT_STRATEGY_BLACK
    self.numthreads: int = self.DEFAULT_NUM_THREADS
    self.brains: dict[PlayerColor, Brain] = {
      PlayerColor.WHITE: self.BRAINS[self.DEFAULT_STRATEGY_WHITE],
      PlayerColor.BLACK: self.BRAINS[self.DEFAULT_STRATEGY_BLACK]
    }
    self.board: Optional[Board] = None

  def __getitem__(self, attr: str):
    return self.__dict__[attr] if attr in self.__dict__ else type(self).__dict__[attr]
  
  def __setitem__(self, attr: str, value: Any):
    self.__dict__[attr] = value

  def start(self) -> None:
    """
    Engine main loop to handle commands.
    """
    self.info()
    while True:
      print(f"ok")
      match input().strip().split():
        case [Command.INFO]:
          self.info()
        case [Command.HELP, *arguments]:
          self.help(arguments)
        case [Command.OPTIONS, *arguments]:
          self.options(arguments)
        case [Command.NEWGAME, *arguments]:
          self.newgame(arguments)
        case [Command.VALIDMOVES]:
          self.validmoves()
        case [Command.BESTMOVE, restriction, value]:
          self.bestmove(restriction, value)
        case [Command.PLAY, move]:
          self.play(move)
        case [Command.PLAY, partial_move_1, partial_move_2]:
          self.play(f"{partial_move_1} {partial_move_2}")
        case [Command.PASS]:
          self.play(Move.PASS)
        case [Command.UNDO, *arguments]:
          self.undo(arguments)
        case [Command.EXIT]:
          print("ok")
          break
        case _:
          self.error("Invalid command. Try 'help' to see a list of valid commands and how to use them")

  def info(self) -> None:
    """
    Handles 'info' command.
    """
    print(f"id HivemindEngine v{self.VERSION}")
    print("Mosquito;Ladybug;Pillbug")

  def help(self, arguments: list[str]) -> None:
    """
    Handles 'help' command with arguments.

    :param arguments: Command arguments.
    :type arguments: list[str]
    """
    if arguments:
      if len(arguments) > 1:
        self.error(f"Too many arguments for command '{Command.HELP}'")
      else:
        match arguments[0]:
          case Command.INFO:
            print(f"  {Command.INFO}")
            print()
            print("  Displays the identifier string of the engine and list of its capabilities.")
            print("  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#info.")
          case Command.HELP:
            print(f"  {Command.HELP}")
            print(f"  {Command.HELP} [Command]")
            print()
            print("  Displays the list of available commands. If a command is specified, displays the help for that command.")
          case Command.OPTIONS:
            print(f"  {Command.OPTIONS}")
            print(f"  {Command.OPTIONS} get OptionName")
            print(f"  {Command.OPTIONS} set OptionName OptionValue")
            print("")
            print("  Displays the available options for the engine. Use 'get' to get the specified OptionName or 'set' to set the specified OptionName to OptionValue.")
            print("  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#options.")
          case Command.NEWGAME:
            print(f"  {Command.NEWGAME} [GameTypeString|GameString]")
            print("")
            print("  Starts a new Base game with no expansion pieces. If GameTypeString is specified, start a game of that type. If a GameString is specified, load it as the current game.")
            print("  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#newgame.")
          case Command.VALIDMOVES:
            print(f"  {Command.VALIDMOVES}")
            print("")
            print("  Displays a list of every valid move in the current game.")
            print("  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#validmoves.")
          case Command.BESTMOVE:
            print(f"  {Command.BESTMOVE} time MaxTime")
            print(f"  {Command.BESTMOVE} depth MaxTime")
            print("")
            print("  Search for the best move for the current game. Use 'time' to limit the search by time in hh:mm:ss or use 'depth' to limit the number of turns to look into the future.")
            print("  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#bestmove.")
          case Command.PLAY:
            print(f"  {Command.PLAY} MoveString")
            print("")
            print("  Plays the specified MoveString in the current game.")
            print("  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#play.")
          case Command.PASS:
            print(f"  {Command.PASS}")
            print("")
            print("  Plays a passing move in the current game.")
            print("  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#pass.")
          case Command.UNDO:
            print(f"  {Command.UNDO} [MovesToUndo]")
            print("")
            print("  Undoes the last move in the current game. If MovesToUndo is specified, undo that many moves.")
            print("  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#undo.")
          case Command.EXIT:
            print(f"  {Command.EXIT}")
            print("")
            print("  Exits the engine.")
          case _:
            self.error(f"Unknown command '{arguments[0]}'")
    else:
      print("Available commands:")
      for command in Command:
        print(f"  {command}")
      print(f"Try '{Command.HELP} <command>' to see help for a particular Command")

  def options(self, arguments: list[str]) -> None:
    """
    Handles 'options' command with arguments.

    :param arguments: Command arguments.
    :type arguments: list[str]
    """
    num_args = len(arguments)
    if not num_args:
      for option in Option:
        self._get_option(option)
    elif num_args >= 2 and num_args <= 3:
      if (option := arguments[1]) in Option:
        if num_args == 2 and arguments[0] == "get":
          self._get_option(Option(option))
        elif num_args == 3 and arguments[0] == "set":
          self._set_option(Option(option), arguments[2])
        else:
          self.error(f"Invalid arguments for command '{Command.OPTIONS}'")
      else:
        self.error(f"Uknown option '{arguments[1]}'")
    else:
      self.error(f"Expected 0, 2, or 3 arguments for command '{Command.OPTIONS}' but got {num_args}")

  def _get_option(self, option: Option) -> None:
    """
    Pretty prints the specified option.

    :param option: Option to print.
    :type option: Option
    """
    print(f"{option};{self.OPTION_TYPES[option]};{self[option.lower()]};{self[f"DEFAULT_{option.name}"]};", end = "")
    match option:
      # Handle options with type Strategy
      case Option.STRATEGY_WHITE | Option.STRATEGY_BLACK:
        print(";".join(Strategy))
      # Handle options with type Int or Float
      case Option.NUM_THREADS:
        print(f"{self[f"MIN_{option.name}"]};{self[f"MAX_{option.name}"]}")
  
  def _set_option(self, option: Option, value: str) -> None:
    """
    Sets the value of the specified option.

    :param option: Option to change.
    :type option: Option
    :param value: New value for the option.
    :type value: str
    """
    valid = True
    match option:
      # Handle options with type Strategy
      case Option.STRATEGY_WHITE | Option.STRATEGY_BLACK if value in Strategy:
        self[option.lower()] = Strategy(value)
        self.brains[PlayerColor[option.name.split("_")[1]]] = self.BRAINS[Strategy(value)]
      # Handle options with type Int
      case Option.NUM_THREADS if value.isdigit() and self[f"MIN_{option.name}"] <= int(value) <= self[f"MAX_{option.name}"]:
        self[option.lower()] = int(value)
      # Handle erroneous use of command
      case _:
        self.error(f"Invalid value for option '{option}'")
        valid = False
    if valid:
      self._get_option(option)

  def newgame(self, arguments: list[str]) -> None:
    """
    Handles 'newgame' command with arguments.  
    Tries to create a new game by instantiating the Board.

    :param arguments: Command arguments.
    :type arguments: list[str]
    """
    try:
      self.board = Board(" ".join(arguments))
      print(self.board)
    except (ValueError, TypeError) as e:
      self.error(e)

  def validmoves(self) -> None:
    """
    Handles 'validmoves' command.
    """
    if self.is_active(self.board):
      print(self.board.valid_moves)

  def bestmove(self, restriction: str, value: str) -> None:
    """
    Handles 'bestmove' command with arguments.

    :param restriction: Type of restriction in searching for the best move.
    :type restriction: str
    :param value: Value of the restriction (time or depth).
    :type value: str
    """
    if self.is_active(self.board):
      if restriction == "time" and re.fullmatch(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]", value):
        print(self.brains[self.board.current_player_color].calculate_best_move(deepcopy(self.board), time_limit = sum(factor * int(time) for factor, time in zip([3600, 60, 1], value.split(':')))))
      elif restriction == "depth" and value.isdigit() and (max_depth := int(value)) > 0:
        print(self.brains[self.board.current_player_color].calculate_best_move(deepcopy(self.board), max_depth = max_depth))
      else:
        self.error(f"Invalid arguments for command '{Command.BESTMOVE}'")

  def play(self, move: str) -> None:
    """
    Handles 'play' command with its argument (MoveString).

    :param move: MoveString.
    :type move: str
    """
    if self.is_active(self.board):
      try:
        self.board.play(move)
        self.brains[self.board.current_player_color].empty_cache()
        print(self.board)
      except ValueError as e:
        self.error(e)

  def undo(self, arguments: list[str]) -> None:
    """
    Handles 'undo' command with arguments.

    :param arguments: Command arguments.
    :type arguments: list[str]
    """
    if self.is_active(self.board):
      if len(arguments) <= 1:
        try:
          if arguments:
            if (amount := arguments[0]).isdigit():
              self.board.undo(int(amount))
            else:
              raise ValueError(f"Expected a positive integer but got '{amount}'")
          else:
            self.board.undo()
          self.brains[self.board.current_player_color].empty_cache()
          print(self.board)
        except ValueError as e:
          self.error(e)
      else:
        self.error(f"Too many arguments for command '{Command.UNDO}'")

  def is_active(self, board: Optional[Board]) -> TypeGuard[Board]:
    """
    Checks whether the current playing Board is initialized.

    :param board: Current playing Board.
    :type board: Optional[Board]
    :return: Whether the Board is initialized.
    :rtype: TypeGuard[Board]
    """
    if not board:
      self.error("No game is currently active")
      return False
    return True

  def error(self, error: str | Exception) -> None:
    """
    Outputs as an error the given message/exception.

    :param message: Message or exception.
    :type message: str | Exception
    """
    print(f"err {error}.")

if __name__ == "__main__":
  Engine().start()
