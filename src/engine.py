from typing import TypeGuard, Final
from enums import Command
from board import Board
from game import Move

VERSION: Final[str] = "1.0.0"
"""
Engine version.
"""

def main() -> None:
  """
  Engine main loop to handle commands.
  """
  info()
  board: Board | None = None
  while True:
    print(f"ok")
    match input().strip().split():
      case [Command.INFO]:
        info()
      case [Command.HELP, *arguments]:
        help(arguments)
      case [Command.OPTIONS]:
        pass
      case [Command.NEWGAME, *arguments]:
        board = newgame(arguments)
      case [Command.VALIDMOVES]:
        validmoves(board)
      case [Command.BESTMOVE, restriction, value]:
        bestmove(board, restriction, value)
      case [Command.PLAY, move]:
        play(board, move)
      case [Command.PLAY, partial_move_1, partial_move_2]:
        play(board, f"{partial_move_1} {partial_move_2}")
      case [Command.PASS]:
        play(board, Move.PASS)
      case [Command.UNDO, *arguments]:
        undo(board, arguments)
      case [Command.EXIT]:
        print("ok")
        break
      case _:
        error("Invalid command. Try 'help' to see a list of valid commands and how to use them")

def info() -> None:
  """
  Handles 'info' command.
  """
  print(f"id HivemindEngine v{VERSION}")
  print("Mosquito;Ladybug;Pillbug")

def help(arguments: list[str]) -> None:
  """
  Handles 'help' command with arguments.

  :param arguments: Command arguments.
  :type arguments: list[str]
  """
  if arguments:
    if len(arguments) > 1:
      error(f"Too many arguments for command '{Command.HELP}'")
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
          error(f"Unknown command '{arguments[0]}'")
  else:
    print("Available commands:")
    for command in Command:
      print(f"  {command}")
    print(f"Try '{Command.HELP} <command>' to see help for a particular Command.")

def newgame(arguments: list[str]) -> Board | None:
  """
  Handles 'newgame' command with arguments.

  :param arguments: Command arguments.
  :type arguments: list[str]
  :return: The newly created Board if successful in creating a new game.
  :rtype: Board | None
  """
  board: Board | None = None
  try:
    board = Board(" ".join(arguments))
    print(board)
  except (ValueError, TypeError) as e:
    error(e)
  return board

def validmoves(board: Board | None) -> None:
  """
  Handles 'validmoves' command.

  :param board: Current playing Board.
  :type board: Board | None
  """
  if is_active(board):
    print(board.valid_moves)

def bestmove(board: Board | None, restriction: str, value: str) -> None:
  """
  Handles 'bestmove' command with arguments.  
  Currently WIP.

  :param board: Current playing Board.
  :type board: Board | None
  :param restriction: Type of restriction in searching the best move.
  :type restriction: str
  :param value: Value of the restriction.
  :type value: str
  """
  if is_active(board):
    # TODO: Check if restriction and value are in the correct format.
    print(board.best_move)

def play(board: Board | None, move: str) -> None:
  """
  Handles 'play' command with its argument (MoveString).

  :param board: Current playing Board.
  :type board: Board | None
  :param move: MoveString.
  :type move: str
  """
  if is_active(board):
    try:
      board.play(move)
      print(board)
    except ValueError as e:
      error(e)

def undo(board: Board | None, arguments: list[str]) -> None:
  """
  Handles 'undo' command with arguments.

  :param board: Current playing Board.
  :type board: Board | None
  :param arguments: Command arguments.
  :type arguments: list[str]
  """
  if is_active(board):
    if len(arguments) <= 1:
      try:
        if arguments:
          if (amount := arguments[0]).isdigit():
            board.undo(int(amount))
          else:
            raise ValueError(f"Expected a positive integer but got '{amount}'")
        else:
          board.undo()
        print(board)
      except ValueError as e:
        error(e)
    else:
      error(f"Too many arguments for command '{Command.UNDO}'")

def is_active(board: Board | None) -> TypeGuard[Board]:
  """
  Checks whether the current playing Board is initialized.

  :param board: Current playing Board.
  :type board: Board | None
  :return: Whether the Board is initialized.
  :rtype: TypeGuard[Board]
  """
  if not board:
    error("No game is currently active")
    return False
  return True

def error(error: str | Exception) -> None:
  """
  Outputs as an error the given message/exception.

  :param message: Message or exception.
  :type message: str | Exception
  """
  print(f"err {error}.")

if __name__ == "__main__":
  main()
