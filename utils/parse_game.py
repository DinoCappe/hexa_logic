import re
import datetime
# 

def parse_sgf(sgf_text):
    prev_metadata = {}
    moves = []

    # --- Extract metadata ---
    meta_pattern = re.compile(r'([A-Z|0-9]{2})\[(.*?)\]')
    meta_matches = meta_pattern.findall(sgf_text[:sgf_text.find('; P0[0')])
    for key, value in meta_matches:
        prev_metadata[key] = value

    result_string = prev_metadata.get("RE")[12:-1]
    assert "draw" not in result_string.lower(), "Draw is not supported"
    if result_string.startswith("guest"):
        result_string = "guest"
    curr_metadata = {
        "GameType": "Base+MLP",
        "Date": datetime.datetime.now().strftime("%Y.%m.%d"),
        "Event": "",
        "Site": "",
        "Round": "",
        "White": prev_metadata.get("P0")[4:-1],
        "Black": prev_metadata.get("P1")[4:-1],
        "Result": "WhiteWins" if result_string == prev_metadata.get("P0")[4:-1] else "BlackWins" 
    }
    # --- Extract moves ---
    move_pattern = re.compile(r'(P[01])\[(\d+)\s+(.*?)\](?:TM\[(\d+)\])?')
    real_turn = 1
    for match in move_pattern.finditer(sgf_text):
        _, _, action, _ = match.groups()
        action = action.strip()
        if action == "Done":
            real_turn += 1
            continue
        values = action.split(' ')
        values[-1] = values[-1].replace("\\\\", "\\")
        values[-1] = values[-1].replace(".", "")
        if action.startswith("Dropb"):
            values[1] = values[1].replace("P1", "P")
            values[1] = values[1].replace("M1", "M")
            values[1] = values[1].replace("L1", "L")
            action = f"{values[1]} {values[-1]}"
        elif action.startswith("Move"):
            values[2] = values[2].replace("P1", "P")
            values[2] = values[2].replace("M1", "M")
            values[2] = values[2].replace("L1", "L")
            action = f"{values[2]} {values[-1]}"
        else:
            continue
        moves.append(f"{real_turn}. {action}")

    return curr_metadata, moves

# --- Main execution ---

if __name__ == '__main__':
    with open('hive_game.sgf', 'r', encoding='utf-8') as f:
        sgf_data = f.read()

    metadata, moves = parse_sgf(sgf_data)

    with open('parsed_game.pgn', 'w', encoding='utf-8') as output_file:
        for key, value in metadata.items():
            output_file.write(f"[{key} \"{value}\"]\n")
        
        output_file.write("\n")

        for move in moves:
            output_file.write(f"{move}\n")
