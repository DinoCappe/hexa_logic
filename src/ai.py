from typing import Optional
from random import choice
from time import sleep
from abc import ABC, abstractmethod
from board import Board

class Brain(ABC):
  """
  Base abstract class for AI agents.
  """

  def __init__(self) -> None:
    self._cache: Optional[str] = None

  @abstractmethod
  def calculate_best_move(self, board: Board) -> str:
    pass

  def empty_cache(self) -> None:
    self._cache = None


class Random(Brain):
  """
  Random acting AI agent.
  """

  def calculate_best_move(self, board: Board) -> str:
      print("[CALC BEST MOVE] Random agent playing")
      valid_moves = [move for move in board.valid_moves.split(";") if move]

      if not valid_moves:
          return "pass"

      # If the cached move is not in the current valid moves, choose a new one.
      if not self._cache or self._cache not in valid_moves:
          self._cache = choice(valid_moves)
      
      return self._cache