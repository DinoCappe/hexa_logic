from typing import Optional, Set, Iterator
from random import choice
from time import sleep
from copy import deepcopy
from abc import ABC, abstractmethod
from enums import GameState
from board import Board

# TODO: Handle time limit
# TODO: Fix alpha-beta pruning.

class Brain(ABC):
  """
  Base abstract class for AI agents.
  """

  def __init__(self) -> None:
    self._cache: Optional[str] = None

  @abstractmethod
  def calculate_best_move(self, board: Board, max_depth: int) -> str:
    """
    Calculates the best move for given board and following the agent policy.

    :param board: Current playing board.
    :type board: Board
    :param max_depth: Maximum lookahead depth.
    :type max_depth: int
    :return: Best move stringified.
    :rtype: str
    """
    pass

  def empty_cache(self) -> None:
    self._cache = None

  def is_terminal(self, node: Board) -> bool:
    return node.state is GameState.DRAW or node.state is GameState.BLACK_WINS or node.state is GameState.WHITE_WINS

class Random(Brain):
  """
  Random acting AI agent.
  """

  def calculate_best_move(self, board: Board, max_depth: int) -> str:
    if not self._cache:
      self._cache = choice(board.valid_moves.split(";"))
    sleep(0.5)
    return self._cache

class AlphaBetaPruner(Brain):
  """
  AI agent following a custom alpha-beta pruning policy.
  """

  def calculate_best_move(self, board: Board, max_depth: int) -> str:
    if not self._cache:
      _, self._cache = self.pruning(board, max_depth)
    return self._cache

  def pruning(self, root: Board, depth: int) -> tuple[float, str]:
    # Stack to simulate recursion
    stack: list[tuple[Board, int, float, float, bool, Iterator[tuple[Board, str]], float, str]] = []
    # Each stack frame contains: (node, depth, alpha, beta, maximizing, child_iter, best_value, best_move)
    stack.append((root, depth, float('-inf'), float('inf'), True, iter(self.gen_children(root)), float('-inf'), root.valid_moves.split(";")[0]))

    final_result = (0, root.valid_moves.split(";")[0])

    while stack:
      node, depth, alpha, beta, maximizing, child_iter, best_value, best_move = stack.pop()

      if depth == 0 or self.is_terminal(node):
        # Evaluate leaf or terminal node
        evaluation = self.evaluate(node)
        if not stack:
          # If this is the root node, store the result
          final_result = (evaluation, best_move)
        else:
          # Pass the evaluation result back up the stack
          stack[-1] = self._update_stack_frame(stack[-1], evaluation, best_move, alpha, beta, maximizing)
        continue

      try:
        # Process the next child
        child, move = next(child_iter)
        stack.append((node, depth, alpha, beta, maximizing, child_iter, best_value, best_move))
        stack.append((child, depth - 1, alpha, beta, not maximizing, iter(self.gen_children(child)), float('-inf') if not maximizing else float('inf'), move))
      except StopIteration:
        # All children processed; finalize the value for this node
        if not stack:
          # If this is the root node, store the result
          final_result = (best_value, best_move)
        else:
          # Pass the result back up the stack
          stack[-1] = self._update_stack_frame(stack[-1], best_value, best_move, alpha, beta, maximizing)

    return final_result

  def _update_stack_frame(self, frame: tuple[Board, int, float, float, bool, Iterator[tuple[Board, str]], float, str], child_value: float, move: str, alpha: float, beta: float, maximizing: bool):
    # Update the stack frame based on the returned child value
    node, depth, _, _, maximizing, child_iter, best_value, best_move = frame

    if maximizing:
      if child_value > best_value:
        best_value, best_move = child_value, move
      alpha = max(alpha, best_value)
    else:
      if child_value < best_value:
        best_value, best_move = child_value, move
      beta = min(beta, best_value)

    if alpha >= beta:
      child_iter = iter([])  # Clear the iterator to skip remaining children (prune)

    return node, depth, alpha, beta, maximizing, child_iter, best_value, best_move

  def evaluate(self, node: Board) -> float:
    minimizing_color = node.current_player_color
    maximizing_color = minimizing_color.opposite

    min_player_surrounding = node.count_queen_neighbors(minimizing_color)
    max_player_surrounding = node.count_queen_neighbors(maximizing_color)

    return min_player_surrounding - max_player_surrounding

  def gen_children(self, parent: Board) -> Set[tuple[Board, str]]:
    return {(deepcopy(parent).play(move), move) for move in parent.valid_moves.split(";")}
