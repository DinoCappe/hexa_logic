import os
import re
from gameWrapper import *
from HiveNNet import NNetWrapper
from enums import PlayerColor

TrainingExample = Tuple[NDArray[np.float64], NDArray[np.float64], float]

class TrainExample:
    def __init__(self, game: GameWrapper, nnet: NNetWrapper):
        self.game = game
        self.nnet = nnet

    def _extract_outcome(self, game_string: str) -> float:
        match = re.search(r'\[Result\s+"(.*?)"\]', game_string)
        if match:
            result = match.group(1)
            if result == "WhiteWins":
                return 1.0
            elif result == "BlackWins":
                return -1.0
            elif result == "Draw":
                return -1e-2  # small bias
        return 0.0
    
    def _extract_extentions(self, game_string: str) -> bool:
        return game_string.find('Base+MLP') != -1
    
    def _parse_game(self, game_string: str) -> List[TrainingExample]:
        board = None
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
            action = board.encode_move_string(line, player)
            pi = np.zeros(self.game.getActionSize(), dtype=np.float64) # Ludo: che ci va qua dentro?
            # pi = self.mcts.getActionProb(board, temp=0)
            symmetries = self.game.getSymmetries(board, pi)
            for b, p in symmetries:
                # b is now an NDArray[np.float64] (the encoded board)
                train_examples.append((b, p, outcome))
                print("Number of training examples generated: ", len(train_examples))
            board, _ = self.game.getNextState(board, player, action, line)
        return train_examples
    
    def train(self, game_string: str):
        train_examples = self._parse_game(game_string)
        self.nnet.train(train_examples)