from typing import List
from datetime import datetime

# Original string
original_string = """"""

# Step 1: split by commas
items = [item.strip() for item in original_string.split(',')]

# Step 2: enumerate and format 
numbered_items: List[str] = []
for idx, item in enumerate(items, start=1):
    numbered_items.append(f"{idx}. {item}")

# Step 3: join all into a single text
output_text = '\n'.join(numbered_items)
header = f"""[GameType "Base"]
[Date "{datetime.now().strftime("%Y.%m.%d")}"]
[Event ""]
[Site ""]
[Round ""]
[White "hexalogic"]
[Black "hexalogic"]
[Result "InProgress"]
"""
filename = f"game_{datetime.now().strftime("%Y%m%d%H%M%S")}.pgn"
with open(filename, 'w') as file:
    file.write(header + output_text)