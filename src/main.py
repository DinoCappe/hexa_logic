import torch
from utils import dotdict
from coach import Coach
from HiveNNet import NNetWrapper
from coach import GameWrapper

def main():
    # Define hyperparameters and configuration.
    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 256,
        'num_layers': 8,
        'mcts_iterations': 100,         # Lower number for testing
        'exploration_constant': 1.41,
        'numEps': 1,                    # Number of self-play games per iteration
        'maxlenOfQueue': 2000,          # Maximum self-play examples to keep in history
        'tempThreshold': 10,            # Temperature threshold for move selection
        'numIters': 2,                  # Number of overall training iterations for testing
        'numItersForTrainExamplesHistory': 20,
        'checkpoint': 'checkpoints',
        'results': 'results'
    })
    
    board_size = (14, 14)
    
    game = GameWrapper()
    action_size = game.getActionSize()
    nnet_wrapper = NNetWrapper(board_size, action_size, args)
    coach = Coach(game, nnet_wrapper, args)
    
    # Run a single self-play episode to generate training examples.
    train_examples = coach.executeEpisode()
    print(f"Self-play episode generated {len(train_examples)} training examples.")
    
    # Now, run the integrated learning loop for a couple of iterations.
    print("Starting integrated learning loop...")
    coach.learn()
    print("Learning loop finished.")

if __name__ == "__main__":
    main()
