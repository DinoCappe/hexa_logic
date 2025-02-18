import math
from copy import deepcopy
from typing import Optional
# from random import choice

from board import Board
from enums import GameState, PlayerColor
from ai import Brain
from game import Move

class MCTSBrain(Brain):
    """
    An MCTS-based AI agent for early game play.
    """
    def __init__(self, iterations: int = 1000, exploration_constant: float = 1.41) -> None:
        super().__init__()
        self.iterations = iterations
        self.C = exploration_constant

    class Node:
        """
        Node in the MCTS search tree.
        """
        def __init__(self, board: Board, parent: Optional['MCTSBrain.Node'] = None, move: Optional[str] = None):
            self.board = board         
            self.parent = parent        
            self.move = move           
            self.children: dict[str, MCTSBrain.Node] = {}
            self.visits = 0             
            self.wins = 0.0             

        def is_terminal(self) -> bool:
            return self.board.state != GameState.IN_PROGRESS and self.board.state != GameState.NOT_STARTED

        def is_fully_expanded(self) -> bool:
            valid_moves = [m for m in self.board.valid_moves.split(";") if m]
            return all(move in self.children for move in valid_moves)

    def calculate_best_move(self, board: Board) -> str:
        """
        Given a board state, runs MCTS for a fixed number of iterations and returns
        the best move (as a move string) based on visit counts.
        """
        root = self.Node(deepcopy(board))
        root_player = board.current_player_color

        for _ in range(self.iterations):
            node = root
            simulation_board = deepcopy(board)

            # --- Selection ---
            # Traverse down the tree using the UCT rule until reaching a node that is not fully expanded
            # or is terminal.
            while not node.is_terminal() and node.is_fully_expanded():
                child = self.select_child(node)
                if child.move is not None:
                    simulation_board.play(child.move)
                node = child

            # --- Expansion ---
            # If the node is non-terminal, expand one of the untried moves.
            if not node.is_terminal():
                move = self.untried_move(node)
                if move is not None:
                    simulation_board.play(move)
                    child = self.Node(deepcopy(simulation_board), parent=node, move=move)
                    node.children[move] = child
                    node = child

            # --- Simulation ---
            # Run a random playout (rollout) from the current simulation board state.
            reward = self.simulate(simulation_board, root_player)

            # --- Backpropagation ---
            # Propagate the reward back up the tree.
            self.backpropagate(node, reward)

        best_move, _ = max(root.children.items(), key=lambda item: item[1].visits)
        return best_move

    def select_child(self, node: 'MCTSBrain.Node') -> 'MCTSBrain.Node':
        best_score = -float('inf')
        best_child: Optional[MCTSBrain.Node] = None
        for child in node.children.values():
            # UCT formula: win rate + exploration term.
            score = (child.wins / child.visits) + self.C * math.sqrt(math.log(node.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            raise Exception("select_child: No valid child found. The node should have at least one child.")
        return best_child

    def untried_move(self, node: 'MCTSBrain.Node') -> Optional[str]:
        """
        Returns a move (as a string) from the board state at this node that has not been tried yet.
        """
        valid_moves = [m for m in node.board.valid_moves.split(";") if m]
        for move in valid_moves:
            if move not in node.children:
                return move
        return None

    def simulate(self, board: Board, root_player: PlayerColor) -> float:
        """
        Runs a random playout on the board until a terminal state is reached.
        Returns the reward for the root player: 1.0 for win, 0.0 for loss, and 0.5 for a draw.
        """
        while board.state == GameState.IN_PROGRESS:
            valid_moves = [m for m in board.valid_moves.split(";") if m]
            move = valid_moves[0] if valid_moves else Move.PASS
            board.play(move)

        if board.state == GameState.DRAW:
            return 0.5
        elif ((board.state == GameState.WHITE_WINS and root_player == PlayerColor.WHITE) or
              (board.state == GameState.BLACK_WINS and root_player == PlayerColor.BLACK)):
            return 1.0
        else:
            return 0.0

    def backpropagate(self, node: Optional['MCTSBrain.Node'], reward: float) -> None:
        """
        Propagates the simulation result (reward) up the tree.
        """
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
