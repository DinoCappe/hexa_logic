import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dotdict
from board import Board
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
import random
import os
import logging

class NNetWrapper:
    def __init__(self, board_size: tuple[int, int], action_size: int, args: dotdict):
        """
        Args:
            board_size: (board_x, board_y) dimensions. For instance, (14, 14).
            action_size: The number of possible moves.
            args: An arguments object with hyperparameters (e.g., num_channels, num_layers, dropout, cuda, etc.).
        """
        self.board_size = board_size
        self.action_size = action_size
        self.args = args
        self.nnet = HiveNNet(board_size, action_size,
                              num_channels=args.num_channels,
                              num_layers=args.num_layers,
                              dropout=args.dropout,
                              )
        if args.cuda:
            self.nnet.cuda()
    
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
    
    def train(self, examples: list[tuple[NDArray[np.float64], NDArray[np.float64], float]]):
        """
        Train the network on a set of examples.
        
        Each example is a tuple:
            (board, target_policy, target_value)
        where:
            board: a Board object,
            target_policy: a vector of move probabilities (numpy array, shape (action_size,)),
            target_value: a scalar value.
        
        This method uses Adam optimizer and trains for a fixed number of epochs.
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['lr'])
        batch_size = self.args['batch_size']
        epochs = self.args['epochs']
        
        self.nnet.train()

        logging.info('STARTING TRAINING')
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            # Shuffle examples
            random.shuffle(examples)
            batch_count = len(examples) // batch_size
            epoch_loss_pi = 0.0
            epoch_loss_v = 0.0
            
            for i in range(batch_count):
                batch = examples[i * batch_size:(i + 1) * batch_size]
                batch_boards, batch_pis, batch_vs = zip(*batch)
                
                # Encode each board using its encode_board method.
                # Ensure each encoded board has shape (4, board_size, board_size)
                encoded_boards = [board.encode_board(grid_size=self.board_size[0]) for board in batch_boards]
                boards_tensor = torch.FloatTensor(np.array(encoded_boards))
                target_pis = torch.FloatTensor(np.array(batch_pis))
                target_vs = torch.FloatTensor(np.array(batch_vs))
                
                if self.args['cuda']:
                    boards_tensor = boards_tensor.cuda()
                    target_pis = target_pis.cuda()
                    target_vs = target_vs.cuda()
                
                out_pi, out_v = self.nnet(boards_tensor)
                # Compute policy loss: negative log likelihood (cross entropy)
                loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                # Compute value loss: Mean squared error
                loss_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size(0)
                total_loss = loss_pi + loss_v
                
                optimizer.zero_grad()
                total_loss.backward() # type: ignore
                optimizer.step() # type: ignore
                
                epoch_loss_pi += loss_pi.item()
                epoch_loss_v += loss_v.item()
            
            avg_loss_pi = epoch_loss_pi / batch_count if batch_count > 0 else float('nan')
            avg_loss_v = epoch_loss_v / batch_count if batch_count > 0 else float('nan')
            msg = f"Epoch {epoch+1}/{self.args.epochs}  Policy Loss = {avg_loss_pi:.4f}, Value Loss = {avg_loss_v:.4f}"
            print(msg)
            logging.info(msg)

    def save_checkpoint(self, folder: str='checkpoint', filename: str='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        torch.save({ # type: ignore
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder: str='checkpoint', filename: str='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location) # type: ignore
        self.nnet.load_state_dict(checkpoint['state_dict'])


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
