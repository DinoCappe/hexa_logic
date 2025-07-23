import torch
from utils import dotdict
from coach import Coach
from HiveNNet import NNetWrapper
from gameWrapper import GameWrapper
from trainExampleWrapper import TrainExampleWrapper
import os
import multiprocessing as mp

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
        'distributed': True,
        'num_channels': 256,
        'num_layers': 8,
        'mcts_iterations': 100,   

        'exploration_constant': 1.41,
        'numEps': 100,                    
        'maxlenOfQueue': 200000,
        'numMCTSSims': 25,          
        'tempThreshold': 15,
        'updateThreshold': 0.55,
        'arenaCompare': 40,             
        'numIters': 100,                 
        'numItersForTrainExamplesHistory': 5,
        'cpuct': 0.8,
        'checkpoint': 'checkpoints',
        'results': 'results'
    })
    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    
    board_size = (14, 14)
    
    game = GameWrapper()
    action_size = game.getActionSize()
    nnet_wrapper = NNetWrapper(board_size, action_size, args)
    if args.distributed:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        nnet_wrapper.nnet = DDP(
            nnet_wrapper.nnet,
            device_ids=[nnet_wrapper.local_rank],
            find_unused_parameters=False
        )
    coach = Coach(game, nnet_wrapper, args)
        
    # Now, run the integrated learning loop for a couple of iterations.
    print("Starting integrated learning loop...")
    coach.learn()
    print("Learning loop finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(True)
