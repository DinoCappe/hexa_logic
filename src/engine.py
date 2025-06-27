from typing import TypeGuard, Final, Optional
from enums import Command
from board import Board, ACTION_SPACE_SIZE
from game import Move
from ai import Brain #, Random
from copy import deepcopy
from mcts import MCTSBrain
from HiveNNet import NNetWrapper
from utils import dotdict
# import time
import torch

class Engine():
  VERSION: Final[str] = "1.1.0"
  """
Engine version.
"""

  def __init__(self) -> None:
    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 256,
        'num_layers': 8,
        'mcts_iterations': 100,   

        'exploration_constant': 1.41,
        'numEps': 2,                    
        'maxlenOfQueue': 5,
        'numMCTSSims': 10,          
        'tempThreshold': 10,
        'updateThreshold': 0.55,
        'arenaCompare': 5,             
        'numIters': 5,                 
        'numItersForTrainExamplesHistory': 20,
        'cpuct': 0,
        'checkpoint': 'checkpoints',
        'results': 'results'
    })

    self.board: Optional[Board] = None
    # self.brain: Brain = Random()
    # TODO: Import a nnet checkpoint from a file if it exists.
    self.nnet = NNetWrapper((14, 14), int(ACTION_SPACE_SIZE+1), args)
    self.brain: Brain = MCTSBrain(nnet=self.nnet, args=args)#iterations=1000)

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
        case [Command.OPTIONS]:
          pass
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
      print(f"Try '{Command.HELP} <command>' to see help for a particular Command.")

  def newgame(self, arguments: list[str]) -> None:
    """
    Handles 'newgame' command with arguments.

    :param arguments: Command arguments.
    :type arguments: list[str]
    :return: The newly created Board if successful in creating a new game.
    :rtype: Optional[Board]
    """
    try:
      self.board = Board(" ".join(arguments))
      print(self.board)
    except (ValueError, TypeError) as e:
      self.error(e)

  def validmoves(self) -> None:
    """
    Handles 'validmoves' command.

    :param board: Current playing Board.
    :type board: Optional[Board]
    """
    if self.is_active(self.board):
      print(self.board.valid_moves)

  def bestmove(self, restriction: str, value: str) -> None:
    """
    Handles 'bestmove' command with arguments.  
    Currently WIP.

    :param board: Current playing Board.
    :type board: Optional[Board]
    :param restriction: Type of restriction in searching the best move.
    :type restriction: str
    :param value: Value of the restriction.
    :type value: str
    """
    if self.is_active(self.board):
      if restriction == "time":
        # current_time = time.time()
        h, m, s = [int(t) for t in value.split(':')]
        seconds = h * 3600 + m * 60 + s
        seconds *= 0.9  # Adjust seconds to allow for some buffer
        print(self.brain.calculate_best_move(deepcopy(self.board), max_time=seconds))
        # Debug: check the real time taken for the move calculation
        # print(f"Time taken: {time.time() - current_time:.2f} seconds")
      else:
        print(self.brain.calculate_best_move(deepcopy(self.board)))

  def play(self, move: str) -> None:
    """
    Handles 'play' command with its argument (MoveString).

    :param board: Current playing Board.
    :type board: Optional[Board]
    :param move: MoveString.
    :type move: str
    """
    if self.is_active(self.board):
      try:
        self.board.play(move)
        self.brain.empty_cache()
        print(self.board)
      except ValueError as e:
        self.error(e)

  def undo(self, arguments: list[str]) -> None:
    """
    Handles 'undo' command with arguments.

    :param board: Current playing Board.
    :type board: Optional[Board]
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
          self.brain.empty_cache()
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
