import math
from copy import deepcopy
from typing import Optional, Dict

from board import Board
from enums import GameState, PlayerColor
from ai import Brain
from game import Move
from HiveNNet import NNetWrapper

class MCTSBrain(Brain):
    """
    An MCTS-based AI agent that integrates a neural network for guiding search.
    """
    def __init__(self, nnet_wrapper: NNetWrapper, iterations: int = 1000, exploration_constant: float = 1.41) -> None:
        super().__init__()
        self.iterations = iterations
        self.C = exploration_constant
        self.nnet_wrapper = nnet_wrapper

    class Node:
        """
        Node in the MCTS search tree.
        """
        def __init__(self, board: Board, parent: Optional['MCTSBrain.Node'] = None, move: Optional[str] = None):
            self.board = board         # The board state at this node.
            self.parent = parent       # Parent node.
            self.move = move           # Move that led to this node.
            self.children: Dict[str, MCTSBrain.Node] = {}
            self.visits = 0
            self.wins = 0.0
            self.priors: Dict[str, float] = {}

        def is_terminal(self) -> bool:
            return self.board.state != GameState.IN_PROGRESS and self.board.state != GameState.NOT_STARTED

        def is_fully_expanded(self) -> bool:
            valid_moves = [m for m in self.board.valid_moves.split(";") if m]
            return all(move in self.children for move in valid_moves)

    def calculate_best_move(self, board: Board) -> str:
        """
        Runs MCTS for a fixed number of iterations and returns the best move (as a move string)
        based on visit counts.
        """
        root = self.Node(deepcopy(board))
        root_player = board.current_player_color

        priors, _ = self.nnet_wrapper.predict(board)
        valid_moves = board.valid_moves.split(";")
        # For simplicity, assume the order of valid_moves aligns with the NN action indices.
        for i, move in enumerate(valid_moves):
            root.priors[move] = priors[i] if i < len(priors) else 0.0

        for _ in range(self.iterations):
            node = root
            simulation_board = deepcopy(board)

            # --- Selection ---
            while not node.is_terminal() and node.is_fully_expanded():
                node = self.select_child(node)
                if node.move is not None:
                    simulation_board.play(node.move)

            # --- Expansion ---
            if not node.is_terminal():
                move = self.untried_move(node)
                if move is not None:
                    simulation_board.play(move)
                    child = self.Node(deepcopy(simulation_board), parent=node, move=move)
                    # Query NN for the new board state.
                    child_priors, _ = self.nnet_wrapper.predict(simulation_board)
                    valid_moves_child = simulation_board.valid_moves.split(";")
                    for i, mv in enumerate(valid_moves_child):
                        child.priors[mv] = child_priors[i] if i < len(child_priors) else 0.0
                    node.children[move] = child
                    node = child

            # --- Simulation ---
            # Use NN's value prediction at the leaf (if not terminal) to avoid lengthy random playouts.
            if not node.is_terminal():
                _, value = self.nnet_wrapper.predict(simulation_board)
            else:
                # If terminal, set reward based on outcome.
                if simulation_board.state == GameState.DRAW:
                    value = 0.5
                elif ((simulation_board.state == GameState.WHITE_WINS and root_player == PlayerColor.WHITE) or
                      (simulation_board.state == GameState.BLACK_WINS and root_player == PlayerColor.BLACK)):
                    value = 1.0
                else:
                    value = 0.0

            # --- Backpropagation ---
            self.backpropagate(node, value)

        best_move, _ = max(root.children.items(), key=lambda item: item[1].visits)
        return best_move

    def select_child(self, node: 'MCTSBrain.Node') -> 'MCTSBrain.Node':
        best_score = -float('inf')
        best_child: Optional[MCTSBrain.Node] = None
        for move, child in node.children.items():
            # PUCT formula: combine win rate and NN prior.
            # Add a small constant to avoid division by zero.
            score = (child.wins / (child.visits + 1e-8)) + self.C * node.priors.get(move, 0) * math.sqrt(node.visits) / (child.visits + 1)
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            raise Exception("select_child: No valid child found. The node should have at least one child.")
        return best_child

    def untried_move(self, node: 'MCTSBrain.Node') -> Optional[str]:
        valid_moves = [m for m in node.board.valid_moves.split(";") if m]
        for move in valid_moves:
            if move not in node.children:
                return move
        return None

    def simulate(self, board: Board, root_player: PlayerColor) -> float:
        """
        Runs a random playout (rollout) until a terminal state is reached.
        As a fallback if NN evaluation isn't used, returns:
            1.0 for win, 0.0 for loss, and 0.5 for draw.
        """
        # For simplicity, we do a random rollout here.
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
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
