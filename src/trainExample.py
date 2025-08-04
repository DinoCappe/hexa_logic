from gameWrapper import *
import re
from HiveNNet import NNetWrapper
from random import shuffle
from arena import Arena
from board import Board, PlayerColor
from mcts import MCTSBrain
import os
import logging

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]

class TrainExample:
    def __init__(self, path: str, game: GameWrapper, nnet: NNetWrapper):
        self.game = game
        self.nnet = nnet
        self.path = path
        self.file_list = os.listdir(path)

    def _extract_outcome(self, game_string: str) -> float:
        match = re.search(r'\[Result\s+"(.*?)"\]', game_string)
        if match:
            result = match.group(1)
            if result == "WhiteWins":
                return 1.0
            elif result == "BlackWins":
                return -1.0
            elif result == "Draw" or result == "AcceptedDraw":
                return -1e-2  # small bias
            elif result == "ResignWhite":
                return -0.95 # bias for resignation
            elif result == "ResignBlack":
                return 0.95 # bias for resignation
        return 0.0
    
    def _extract_extentions(self, game_string: str) -> bool:
        return game_string.find('Base+MLP') != -1
    
    def parse_game(self, game_string: str) -> List[TrainingExample]:
        if self._extract_extentions(game_string):
            board = self.game.getInitBoard(expansions=True)
        else:
            board = self.game.getInitBoard()

        player = 1 if board.current_player_color == PlayerColor.WHITE else 0
        outcome = self._extract_outcome(game_string)
        game_string = game_string[game_string.find('\n\n') + 2:]  # Skip the header
        lines = [line for line in game_string.split('\n') if line.strip()]
        train_examples: list[TrainingExample] = []

        action_size = self.game.getActionSize()

        for line in lines:
            line = line[line.find('.') + 2:]  # Skip the move number

            # Encode the human move; if it fails, abort this game.
            try:
                action = board.encode_move_string(line)
            except Exception as e:
                print(f"Failed to encode human move '{line}': {e}. Skipping game.")
                return []

            canon = board if board.current_player_color == PlayerColor.WHITE else board.invert_colors()

            # One-hot policy for the human move.
            pi = np.zeros(action_size, dtype=np.float64)
            pi[action] = 1.0

            # Value from the perspective of the player to move.
            value = outcome if board.current_player_color == PlayerColor.WHITE else -outcome

            # Apply symmetries.
            symmetries = self.game.getSymmetries(canon, pi)
            for b, p in symmetries:
                train_examples.append((b, p, value))
                print("Number of training examples generated: ", len(train_examples))

            # Step the real board forward and update player.
            try:
                board, player = self.game.getNextState(board, player, action, line)
            except Exception as e:
                print(f"Failed to apply human move '{line}' to board: {e}. Stopping parsing this game.")
                break

        return train_examples

    def execute_training(self):
        train_examples = []
        for file in self.file_list:
            if file.endswith('.pgn'):
                input_path = os.path.join(self.path, file)
                with open(input_path, 'r') as file:
                    content = file.read()
                    train_examples.extend(self.parse_game(content))
            if len(train_examples) > 20000:
                shuffle(train_examples)
                self.nnet.train(train_examples)
                train_examples = []

    def mcts_agent_action(self, board: Board, player: int) -> int:
        """
        Wraps the *current* network (self.nnet) so it looks like a
        Callable[[Board,int],int].
        """
        mcts = MCTSBrain(self.game, self.nnet, self.nnet.args)
        pi = mcts.getActionProb(board, temp=0)
        valid = self.game.getValidMoves(board)
        pi = pi * valid
        if pi.sum() > 0:
            return int(np.argmax(pi))
        else:
            return int(np.random.choice(np.nonzero(valid)[0]))
    
    def random_agent_action(self, board: Board, player: int) -> int:
        """
        Ignores canonicalisation completely—
        just pick uniformly from the real board’s valid moves.
        """
        valid = self.game.getValidMoves(board)
        valid_indices = np.nonzero(valid)[0]
        return int(np.random.choice(valid_indices))

    def evaluate_training(self):
        arena = Arena(self.random_agent_action,
                self.mcts_agent_action,
                self.game)
        wins_random, wins_zero, draws = arena.playGames(10)
        print('ZERO/RANDOM WINS : %d / %d ; DRAWS : %d', wins_zero, wins_random, draws)
        logging.info(f'ZERO/RANDOM WINS : {wins_zero} / {wins_random} ; DRAWS : {draws}')