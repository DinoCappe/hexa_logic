import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import dotdict
from board import Board

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
        
        # Return probabilities (by exponentiating the log probabilities) and value.
        return torch.exp(pi).detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]  # type: ignore


class HiveNNet(nn.Module):
    def __init__(self, board_size: tuple[int, int], action_size: int, 
                 num_channels: int=256, num_layers: int=4, dropout: float=0.3):
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
        
        # Set input channels based on the chosen representation.
        in_channels = 4

        # First convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        
        # Additional convolutional layers.
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.convs.append(nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=1))
            self.bns.append(nn.BatchNorm2d(self.num_channels))
        
        # Here we assume no change in spatial dimensions (stride=1, padding=1).
        flattened_size = self.num_channels * self.board_x * self.board_y
        
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


# Example usage:
if __name__ == "__main__":
    board_size = (14, 14)
    action_size = 100  # Replace with your actual action count.
    
    args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,
    'num_layers': 4
    })
    
    # Instantiate the network wrapper.
    nnet_wrapper = NNetWrapper(board_size, action_size, args)
    
    dummy_board = Board()
    dummy_encoded_board = np.zeros((4, board_size[0], board_size[1]), dtype=np.int32)
    
    # For testing, we'll mimic predict() by using our dummy board encoding.
    try:
        pi, v = nnet_wrapper.predict(dummy_board)
        print("Policy output shape:", pi.shape)
        print("Value output shape:", v.shape)
    except Exception as e:
        print("Error during prediction:", e)
