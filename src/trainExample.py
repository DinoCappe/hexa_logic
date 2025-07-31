from gameWrapper import *
import re
from HiveNNet import NNetWrapper
from random import shuffle
from arena import Arena
from board import Board, PlayerColor
from mcts import MCTSBrain
import os

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
        board = None
        mcts = MCTSBrain(self.game, self.nnet, self.nnet.args)
        if self._extract_extentions(game_string):
            board = self.game.getInitBoard(expansions=True)
        else:
            board = self.game.getInitBoard()
        player = 1 if board.current_player_color == PlayerColor.WHITE else 0
        outcome = self._extract_outcome(game_string)
        game_string = game_string[game_string.find('\n\n') + 2:]  # Skip the header
        # Loop through game_string every newline
        lines = [line for line in game_string.split('\n') if line.strip()]
        train_examples = []
        for line in lines:
            line = line[line.find('.') + 2:] # Skip the move number
            action = board.encode_move_string(line)
            # Va invertita la board?
            # canon = copy.deepcopy(board)
            # if canon.current_player_color == PlayerColor.BLACK:
            #     canon = board.invert_colors()
            # La ricerca dopo un po' fallisce in una mossa non valida
            pi = mcts.getActionProb(board)
            symmetries = self.game.getSymmetries(board, pi)
            for b, p in symmetries:
                # b is now an NDArray[np.float64] (the encoded board)
                train_examples.append((b, p, outcome))
                print("Number of training examples generated: ", len(train_examples))
            board, _ = self.game.getNextState(board, player, action, line)
        return train_examples
        
    def execute_training(self):
        train_examples = []
        for file in self.file_list:
            if file.endswith('.pgn'):
                input_path = os.path.join(self.path, file)
                with open(input_path, 'r') as file:
                    content = file.read()
                    train_examples.extend(self.parse_game(content))
        shuffle(train_examples)
        self.nnet.train(train_examples)

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