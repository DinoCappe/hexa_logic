import torch
from utils import dotdict
from coach import Coach
from HiveNNet import NNetWrapper
from gameWrapper import GameWrapper
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

def main():
    # Define hyperparameters and configuration.
    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 128,
        'cuda': torch.cuda.is_available(),
        'distributed': True,
        'train_ddp': False,
        'num_channels': 256,
        'num_layers': 8,
        'mcts_iterations': 100,   

        'exploration_constant': 1.41,
        'numEps': 60,                    
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
    
    board_size = (26, 26)
    
    game = GameWrapper()
    action_size = game.getActionSize()
    nnet_wrapper = NNetWrapper(board_size, action_size, args)

    # --- load pretrained weights if present ---
    pretrain_ckpt = os.path.join(args.checkpoint, "pretrain_last.pth.tar")
    if os.path.isfile(pretrain_ckpt):
        try:
            nnet_wrapper.load_checkpoint(folder=args.checkpoint, filename="pretrain_last.pth.tar")
            print("[main] loaded pretrained checkpoint:", pretrain_ckpt)
        except Exception as e:
            print("[main] WARNING: failed to load pretrained checkpoint:", e)

    if args.distributed:
        import torch.distributed as dist
        is_rank0 = (dist.get_rank() == 0)
    else:
        is_rank0 = True

    if is_rank0:
        nnet_wrapper.save_checkpoint(folder=args.checkpoint, filename="best.pth.tar")
        nnet_wrapper.save_checkpoint(folder=args.checkpoint, filename="temp.pth.tar")
        print("[main] seeded best.pth.tar and temp.pth.tar from pretrained weights")

    if args.distributed:
        dist.barrier()

    if args.distributed and args.train_ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        nnet_wrapper.nnet = DDP(
            nnet_wrapper.nnet,
            device_ids=[nnet_wrapper.local_rank],
            find_unused_parameters=False
        )
        nnet_wrapper.ddp_wrapped = True
    coach = Coach(game, nnet_wrapper, args)
        
    print("Starting integrated learning loop...")
    coach.learn()
    print("Learning loop finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
