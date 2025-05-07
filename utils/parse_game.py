import re
import datetime
import os

def parse_sgf(sgf_text):
    prev_metadata = {}
    moves = []

    # --- Extract metadata ---
    meta_pattern = re.compile(r'([A-Z|0-9]{2})\[(.*?)\]')
    meta_matches = meta_pattern.findall(sgf_text[:sgf_text.find('; P0[0')])
    for key, value in meta_matches:
        prev_metadata[key] = value
    game_type = ""
    if prev_metadata["SU"].startswith("hive-plm"):
        game_type = "Base+MLP"
    else:
        game_type = "Base"

    result_string = prev_metadata.get("RE")
    draw = result_string == "The game is a draw"
    check_result = prev_metadata.get("P0")[4:-1] in prev_metadata.get("P1")[4:-1]
    last_player = None

    curr_metadata = {
        "GameType": game_type,
        "Date": datetime.datetime.now().strftime("%Y.%m.%d"),
        "Event": "",
        "Site": "",
        "Round": "",
        "White": prev_metadata.get("P0")[4:-1],
        "Black": prev_metadata.get("P1")[4:-1],
        "Result": "WhiteWins" if prev_metadata.get("P0")[4:-1] in result_string else "BlackWins" if not draw else "Draw"
    }
    # --- Extract moves ---
    move_pattern = re.compile(r'(P[01])\[(\d+)\s+(.*?)\](?:TM\[(\d+)\])?')
    real_turn = 1
    for match in move_pattern.finditer(sgf_text):
        player, _, action, _ = match.groups()
        action = action.strip()
        if action == "Done":
            real_turn += 1
            continue
        values = action.split(' ')
        values[-1] = values[-1].replace("\\\\", "\\")
        values[-1] = values[-1].replace(".", "")
        if values[0] == "Drop":
            return {}, []
        if values[0] == "Dropb" or values[0] == "Pdropb":
            values[1] = values[1].replace("P1", "P")
            values[1] = values[1].replace("M1", "M")
            values[1] = values[1].replace("L1", "L")
            if values[-1] == "":
                action = f"{values[1]}"
            else:
                action = f"{values[1]} {values[-1]}"
        elif values[0] == "Move" or values[0] == "PMove":
            values[2] = values[2].replace("P1", "P")
            values[2] = values[2].replace("M1", "M")
            values[2] = values[2].replace("L1", "L")
            if values[-1] == "":
                action = f"{values[2]}"
            else:
                action = f"{values[2]} {values[-1]}"
        elif values[0] == "Pass":
            action = "pass"
        elif values[0] == "Pick" or values[0] == "Pickb" or values[0] == "Start":
            continue
        # TODO: Handle "AcceptDraw" and "Resign" actions
        elif values[0] == "OfferDraw" or values[0] == "AcceptDraw" or values[0] == "Resign" or values[0] == "DeclineDraw":
            real_turn -= 1
            continue
        else:
            raise ValueError(f"Unknown action: {action}")
        moves.append(f"{real_turn}. {action}")
        last_player = player

    if check_result:
        if last_player == "P0":
            curr_metadata["Result"] = "WhiteWins"
        else:
            curr_metadata["Result"] = "BlackWins"

    return curr_metadata, moves

# --- Main execution ---

if __name__ == '__main__':
    input_directory = 'games'
    output_directory = 'UHP_games'

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        # Not clear what the "U" and "A" prefixes are for
        if filename.endswith('.sgf'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.pgn")

            with open(input_path, 'r', encoding='utf-8') as f:
                sgf_data = f.read()

            metadata, moves = parse_sgf(sgf_data)

            if not metadata or not moves:
                print(f"Unable to parse {filename}, skipping...")
                continue

            with open(output_path, 'w', encoding='utf-8') as output_file:
                for key, value in metadata.items():
                    output_file.write(f"[{key} \"{value}\"]\n")
                
                output_file.write("\n")

                for move in moves:
                    output_file.write(f"{move}\n")
        
            
