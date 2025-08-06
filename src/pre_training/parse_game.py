import re
import datetime
import os
import sys
from pathlib import Path
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from board import Board
from enums import GameState

def parse_sgf(sgf_text):
    prev_metadata = {}
    moves = []

    # --- Extract metadata ---
    meta_pattern = re.compile(r'([A-Z|0-9]{2})\[(.*?)\]')
    meta_matches = meta_pattern.findall(sgf_text[:sgf_text.find('; P0[0')])
    for key, value in meta_matches:
        prev_metadata[key] = value
    game_type = ""
    su = prev_metadata.get("SU", "").strip().lower()
    if su.startswith("hive-plm"):
        game_type = "Base+MLP"
    elif su.startswith("hive "):
        game_type = "Base"
    else:
        print("Hybrid game type not supported")
        return {}, []


    curr_metadata = {
        "GameType": game_type,
        "Date": datetime.datetime.now().strftime("%Y.%m.%d"),
        "Event": "",
        "Site": "",
        "Round": "",
        "White": prev_metadata.get("P0")[4:-1],
        "Black": prev_metadata.get("P1")[4:-1],
        "Result": ""
    }
    
    # --- Extract moves ---
    move_pattern = re.compile(r'(P[01])\[(\d+)\s+(.*?)\](?:TM\[(\d+)\])?')
    real_turn = 0
    for match in move_pattern.finditer(sgf_text):
        player, _, action, _ = match.groups()
        action = action.strip()
        if action.lower() == "done":
            real_turn += 1
            continue
        values = action.split(' ')
        values[0] = values[0].lower()   # <--- normalize action type
        values[-1] = values[-1].replace("\\\\", "\\")
        values[-1] = values[-1].replace(".", "")
        if values[0] == "drop":
            return {}, []
        elif values[0] in ("dropb", "pdropb"):
            values[1] = values[1].replace("P1", "P")
            values[1] = values[1].replace("M1", "M")
            values[1] = values[1].replace("L1", "L")
            if values[-1] == "":
                if real_turn > 0:
                    return {}, []
                action = f"{values[1]}"
            else:
                action = f"{values[1]} {values[-1]}"
        elif values[0] in ("move", "pmove"):
            values[2] = values[2].replace("P1", "P")
            values[2] = values[2].replace("M1", "M")
            values[2] = values[2].replace("L1", "L")
            if values[-1] == "":
                if real_turn > 0:
                    return {}, []
                action = f"{values[2]}"
            else:
                action = f"{values[2]} {values[-1]}"
        elif values[0] == "pass":
            action = "pass"
        elif values[0] in ("pick", "pickb", "start"):
            continue
        elif values[0] == "acceptdraw":
            curr_metadata["Result"] = "AcceptedDraw"
            return curr_metadata, moves
        elif values[0] == "resign":
            if player == "P0":
                curr_metadata["Result"] = "ResignWhite"
            else:
                curr_metadata["Result"] = "ResignBlack"
            return curr_metadata, moves
        elif values[0] in ("offerdraw", "declinedraw"):
            real_turn -= 1
            continue
        else:
            print(f"Unknown action type: {values[0]}")
            return {}, []
        if len(moves) <= real_turn:
            moves.append(action)
        else:
            moves[real_turn] = action

    return curr_metadata, moves

def check_game(metadata, moves):
    """
    Check if the game is valid based on the branching factor and other criteria.
    This is a placeholder function; actual implementation may vary.
    """
    board = Board(metadata["GameType"])
    avg_branching_factor = 0
    for i, move in enumerate(moves):
        avg_branching_factor += len(board._get_valid_moves())
        try:
            board.play(move)
        except ValueError as e:
            if board.state == GameState.IN_PROGRESS and "pass" in board.valid_moves and ('bot' in metadata["White"].lower() or 'bot' in metadata["Black"].lower()):
                board.play("pass")
                moves.insert(i, "pass")
                print(f"Adding pass after {move} {i+1} for bot game")
            else:
                print(f"Error playing move {move} {i+1}: {e}\n game: {moves}\n board: {board.state}")
                return False, 0.0
    avg_branching_factor /= len(moves)
    match board.state:
        case GameState.WHITE_WINS:
            metadata["Result"] = "WhiteWins"
        case GameState.BLACK_WINS:
            metadata["Result"] = "BlackWins"
        case GameState.DRAW:
            metadata["Result"] = "Draw"
        case _:
            if metadata["Result"] != "AcceptedDraw" and not metadata["Result"].startswith("Resign"):
                print("Game not finished or invalid state.")
                return False, avg_branching_factor
    return True, avg_branching_factor

# --- Main execution ---

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Parse all .sgf in one subfolder of games/ into UHP_games/"
    )
    parser.add_argument(
        'year',
        help="Which subfolder of games/ to process",
    )
    args = parser.parse_args()

    input_directory = base_dir / "games" / args.year
    output_directory = base_dir / "UHP_games" / args.year
    output_analysis = base_dir / "game_analysis.txt"
    blacklist = ['HV-guest-SmartBot-2025-02-02-1535.sgf', 'HV-Snowulf-Dumbot-2024-12-30-1455.sgf', 'HV-Steevee-WeakBot-2025-02-01-2031.sgf', 'HV-SmartBot-guest-2025-01-06-0837.sgf', 'HV-SmartBot-guest-2025-04-19-0939.sgf', 'HV-WeakBot-guest-2025-05-03-1954.sgf', 'HV-whtiger-Dumbot-2025-02-12-2027.sgf']

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        # Not clear what the "U" and "A" prefixes are for
        if filename.endswith('.sgf') and filename not in blacklist:
            input_path  = input_directory  / filename
            output_path = output_directory / f"{filename[:-4]}.pgn"

            with open(input_path, 'r', encoding='utf-8') as f:
                sgf_data = f.read()

            metadata, moves = parse_sgf(sgf_data)

            if not metadata or not moves:
                print(f"Unable to parse {filename}, skipping...")
                continue

            if len(moves) < 20 or len(moves) > 110:
                print(f"Game {filename} has too few or too many moves, skipping...")
                continue
            
            is_valid, avg_branching_factor = check_game(metadata, moves)

            if is_valid:
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    for key, value in metadata.items():
                        output_file.write(f"[{key} \"{value}\"]\n")
                    
                    output_file.write("\n")

                    for i, move in enumerate(moves):
                        output_file.write(f"{i+1}. {move}\n")
            else:
                if avg_branching_factor == 0.0:
                    print(f"Invalid game in {filename}, skipping...")
                    # input("Press Enter to continue...")

