import torch
from utils import dotdict
from board import Board
from enums import GameState
from mcts import MCTSBrain
from HiveNNet import NNetWrapper

def play_full_game():
    board = Board()
    print("Initial board:")
    print(board)
    
    board_size = (14, 14)
    action_size = 100
    
    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 256,
        'num_layers': 4,
        'results': 'experiment_results'
    })
    
    nnet_wrapper = NNetWrapper(board_size, action_size, args)
    mcts = MCTSBrain(nnet_wrapper, iterations=100, exploration_constant=1.41)
    
    move_count = 0
    max_moves = 20  # Set a limit to prevent infinite loops during testing.
    
    while board.state in [GameState.NOT_STARTED, GameState.IN_PROGRESS] and move_count < max_moves:
        if board.state == GameState.NOT_STARTED:
            valid_moves = board.valid_moves.split(";")
            if valid_moves:
                board.play(valid_moves[0])
                print("\nBoard after initial move:")
                print(board)
        else:
            best_move = mcts.calculate_best_move(board)
            print(f"\nBest move found at move {move_count+1}: {best_move}")
            board.play(best_move)
            move_count += 1
            print(f"\nBoard after move {move_count}:")
            print(board)
            print("Valid moves:", board.valid_moves)
    print("\nGame ended with state:", board.state)

if __name__ == "__main__":
    play_full_game()
