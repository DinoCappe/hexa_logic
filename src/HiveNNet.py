import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dotdict
from board import Board
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
import os
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import multiprocessing as mp

def setup_distributed():
    """Setup distributed training with proper error handling"""
    try:
        # 1) figure out which local rank we are
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 2) bind this process to the right GPU *before* init
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # 3) now start the process group
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        return local_rank
    except Exception as e:
        logging.error(f"Failed to setup distributed training: {e}")
        return 0

class SelfPlayDataset(Dataset):
    def __init__(self, examples: list[tuple[NDArray[np.float64], NDArray[np.float64], float]], board_size: int):
        self.boards, self.pis, self.zs = zip(*examples)
        # pre‐encode any Board objects into raw numpy arrays
        self.encoded = []
        for b in self.boards:
            if isinstance(b, np.ndarray):
                self.encoded.append(b)
            else:
                # b is a Board instance
                self.encoded.append(b.encode_board(grid_size=board_size))
        self.encoded = np.stack(self.encoded)              # shape (N,C,H,W)
        self.pis = np.stack(self.pis)                  # shape (N,action_size)
        self.zs = np.array(self.zs, dtype=np.float32) # shape (N,)

    def __len__(self):
        return len(self.zs)

    def __getitem__(self, idx):
        # return torch tensors
        return (
            torch.from_numpy(self.encoded[idx]).float(),
            torch.from_numpy(self.pis[idx]).float(),
            torch.tensor(self.zs[idx]).float(),
        )

class NNetWrapper:
    def __init__(self, board_size: tuple[int, int], action_size: int, args: dotdict):
        """
        Args:
            board_size: (board_x, board_y) dimensions. For instance, (14, 14).
            action_size: The number of possible moves.
            args: An arguments object with hyperparameters (e.g., num_channels, num_layers, dropout, cuda, etc.).
        """
        self.args = args
        self.distributed = args.distributed

        if self.distributed:
            self.local_rank = setup_distributed()
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.device = torch.device("cuda" if args.cuda else "cpu")
        
        self.board_size = board_size
        self.action_size = action_size
        self.nnet = HiveNNet(board_size, action_size,
            num_channels=args.num_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            )
        
        self.nnet.to(self.device)
    
    def predict(self, board: Board):
        """
        Given a Board, encode it and return the network's output:
          - A policy vector (as log probabilities)
          - A value estimate (scalar)
        """
        grid_size = self.board_size[0]  # assuming a square board
        encoded_board = board.encode_board(grid_size=grid_size)
        
        # Add a batch dimension.
        input_tensor = torch.FloatTensor(encoded_board).unsqueeze(0)
        if self.args.cuda:
            input_tensor = input_tensor.cuda()
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(input_tensor)
        
        return torch.exp(pi).detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]  # type: ignore
    
    def train(self, examples):
        """
        Train the network on a set of examples with improved memory management
        and error handling for distributed training.
        """
        if not examples:
            logging.warning("No training examples provided, skipping training")
            return
            
        try:
            optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['lr'])
            batch_size = self.args['batch_size']
            epochs = self.args['epochs']

            # --- create the dataset and loader
            ds = SelfPlayDataset(examples, board_size=self.board_size[0])
            num_examples = len(ds)
            
            if self.distributed:
                sampler = DistributedSampler(ds, shuffle=False)
            else:
                sampler = None

            # Reduce num_workers for better memory management
            num_workers = min(2, mp.cpu_count() // 2) if self.distributed else 4
            
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True  # Helps with distributed training
            )
            num_batches = len(loader)

            # Log overall dataset/loader stats
            rank = dist.get_rank() if self.distributed else 0
            logging.info(f"[R{rank}] STARTING TRAINING on {num_examples} examples, {num_batches} batches per epoch, {epochs} epochs")
            print(f"[R{rank}][TRAIN] examples={num_examples}, batches/epoch={num_batches}, epochs={epochs}")

            self.nnet.train()

            for epoch in range(1, epochs+1):
                if self.distributed and sampler:
                    sampler.set_epoch(epoch)

                epoch_loss_pi = 0.0
                epoch_loss_v = 0.0
                nbatches = 0

                logging.info(f"[R{rank}]  → Epoch {epoch}/{epochs} starting")
                print(f"[R{rank}][TRAIN] Epoch {epoch}/{epochs} ...")

                for boards_tensor, target_pis, target_vs in loader:
                    try:
                        boards_tensor = boards_tensor.to(self.device, non_blocking=True)
                        target_pis = target_pis.to(self.device, non_blocking=True)
                        target_vs = target_vs.to(self.device, non_blocking=True)

                        out_pi, out_v = self.nnet(boards_tensor)
                        loss_pi = - (target_pis * out_pi).sum(dim=1).mean()
                        loss_v  = (target_vs - out_v.view(-1)).pow(2).mean()
                        loss = loss_pi + loss_v

                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=1.0)
                        
                        optimizer.step()

                        epoch_loss_pi += loss_pi.item()
                        epoch_loss_v += loss_v.item()
                        nbatches += 1
                        
                        # Clear cache periodically to prevent OOM
                        if nbatches % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logging.error(f"[R{rank}] CUDA OOM in training batch {nbatches}, skipping...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                if nbatches > 0:
                    avg_pi = epoch_loss_pi / nbatches
                    avg_v  = epoch_loss_v  / nbatches
                    msg = f"[R{rank}] Epoch {epoch}/{epochs} complete — avg π loss={avg_pi:.4f}, avg v loss={avg_v:.4f}"
                    logging.info(msg)
                    print(msg)
                else:
                    logging.warning(f"[R{rank}] No successful batches in epoch {epoch}")
                    
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise e

    def save_checkpoint(self, folder: str='checkpoint', filename: str='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)

        # If the model is wrapped by DDP, pull out the underlying module
        if hasattr(self.nnet, 'module'):
            state_dict = self.nnet.module.state_dict()
        else:
            state_dict = self.nnet.state_dict()

        torch.save({'state_dict': state_dict}, filepath)

    def load_checkpoint(self, folder: str='checkpoint', filename: str='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)  # type: ignore
        state_dict = checkpoint['state_dict']

        # If we're wrapped by DDP, load into the underlying module;
        # otherwise load directly.
        if hasattr(self.nnet, 'module'):
            self.nnet.module.load_state_dict(state_dict)
        else:
            self.nnet.load_state_dict(state_dict)


class HiveNNet(nn.Module):
    def __init__(self, board_size: tuple[int, int], action_size: int, 
                 num_channels: int=256, num_layers: int=8, dropout: float=0.3):
        """
        Args:
            board_size (tuple): (board_x, board_y) dimensions.
            action_size (int): Number of possible moves.
            num_channels (int): Number of convolutional filters.
            num_layers (int): Number of convolutional layers (supported: 4, 6, 8).
            dropout (float): Dropout rate.
        """
        super().__init__()  # type: ignore
        self.board_x, self.board_y = board_size
        self.action_size = action_size
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        in_channels = 4

        # First convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        # Second convolutional layer.
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        
        # Additional convolutional layers.
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _i in range(2, self.num_layers):
            self.convs.append(nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1))
            self.bns.append(nn.BatchNorm2d(self.num_channels))
        
        # conv1 and conv2 preserve the dimensions.
        # Additional layers reduce spatial dimensions by 2 per layer.
        num_extra_layers = self.num_layers - 2  # For self.num_layers=8, that's 6 extra layers.
        reduction = 2 * num_extra_layers         # 6*2 = 12.
        effective_size = self.board_x - reduction  # 14 - 12 = 2 (assuming square input)
        flattened_size = self.num_channels * effective_size * effective_size  # 256 * 2 * 2 = 1024
        
        # Fully connected layers.
        self.fc1 = nn.Linear(flattened_size, 4096)
        self.fc_bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc_bn2 = nn.BatchNorm1d(2048)
        
        # Policy head.
        self.fc_policy = nn.Linear(2048, self.action_size)
        # Value head.
        self.fc_value = nn.Linear(2048, 1)

    def forward(self, s: torch.Tensor):
        """
        Forward pass.
        Args:
            s (torch.Tensor): Input board tensor. Its shape should be 
                              (batch_size, board_x, board_y) for simple representations,
                              or include channel information for spatial planes.
        Returns:
            (pi, v): where pi is the log-softmax over actions and v is the board value.
        """
        batch_size = s.size(0)
        in_channels = 4
        s = s.view(batch_size, in_channels, self.board_x, self.board_y)
        
        # Convolutional layers.
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        for conv, bn in zip(self.convs, self.bns):
            s = F.relu(bn(conv(s)))
        
        # Flatten.
        s = s.view(batch_size, -1)
        
        # Fully connected layers.
        s = F.relu(self.fc_bn1(self.fc1(s)))
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = F.relu(self.fc_bn2(self.fc2(s)))
        s = F.dropout(s, p=self.dropout, training=self.training)
        
        # Policy head.
        pi = self.fc_policy(s)
        pi = F.log_softmax(pi, dim=1)
        
        # Value head.
        v = self.fc_value(s)
        v = torch.tanh(v)
        
        return pi, v
