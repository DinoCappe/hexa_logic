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
    if not self._cache:
      self._cache = choice(board.valid_moves.split(";"))
    sleep(0.5)
    return self._cache

class AlphaBetaPruner(Brain):
  """
  AI agent following a custom alpha-beta pruning policy.
  """

  def calculate_best_move(self, board: Board) -> str:
    if not self._cache:
      self._cache = choice(board.valid_moves.split(";"))
    return self._cache
