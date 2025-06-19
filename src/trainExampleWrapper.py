from trainExample import TrainExample
from gameWrapper import GameWrapper
from HiveNNet import NNetWrapper
import os

class TrainExampleWrapper:
    def __init__(self, path: str, game: GameWrapper, nnet: NNetWrapper):
        self.train_examples = TrainExample(game, nnet)
        self.path = path
        self.file_list = os.listdir(path)
        
    def execute_training(self):
        for file in self.file_list:
            if file.endswith('.pgn'):
                input_path = os.path.join(self.path, file)
                with open(input_path, 'r') as file:
                    content = file.read()
                try:
                    _ = self.train_examples._parse_game(content)
                except ValueError as e:
                    print(f"Error parsing {file}: {e}")
                    return
                # self.train_examples.train(content)