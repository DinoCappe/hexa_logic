import torch
from utils import dotdict
from coach import Coach
from HiveNNet import NNetWrapper
from gameWrapper import GameWrapper
from trainExampleWrapper import TrainExampleWrapper

import logging

logging.basicConfig(
    level=logging.INFO,
    filename='log_file.log',
    filemode='a',
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

log = logging.getLogger(__name__)

def main(pre_training: bool = False):
    # Define hyperparameters and configuration.
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
    
    board_size = (14, 14)
    
    game = GameWrapper()
    action_size = game.getActionSize()
    nnet_wrapper = NNetWrapper(board_size, action_size, args)
    if pre_training:
        train_wrapper = TrainExampleWrapper('utils/UHP_games', game, nnet_wrapper)
        train_wrapper.execute_training()
        # TODO: Implement filtering logic based on brenching factor and other criteria
    else:
        coach = Coach(game, nnet_wrapper, args)
        
        # Run a single self-play episode to generate training examples.
        train_examples = coach.executeEpisode()
        print(f"Self-play episode generated {len(train_examples)} training examples.")
        
        # Now, run the integrated learning loop for a couple of iterations.
        print("Starting integrated learning loop...")
        coach.learn()
        print("Learning loop finished.")

if __name__ == "__main__":
    main(True)
