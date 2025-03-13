# Canonical dictionary: ignore color and simply assign a unique code for each bug type.
TILE_DICT_CANONICAL = {
    "wQ": 0,  # White pieces
    "wA1": 1,
    "wA2": 2,
    "wA3": 3,
    "wG1": 4,
    "wG2": 5,
    "wG3": 6,
    "wB1": 7,
    "wB2": 8,
    "wS1": 9,
    "wS2": 10,
    "wM": 11,
    "wL": 12, 
    "wP": 13,  
    "bQ": 0,  # Dark pieces
    "bA1": 1,
    "bA2": 2,
    "bA3": 3,
    "bG1": 4,
    "bG2": 5,
    "bG3": 6,
    "bB1": 7,
    "bB2": 8,
    "bS1": 9,
    "bS2": 10,
    "bM": 11,
    "bL": 12,
    "bP": 13,
}

# Inverse canonical mapping: maps a canonical tile code (0–13) to the bug type string.
INV_TILE_DICT_CANONICAL = {
    0: "Q",
    1: "A1",
    2: "A2",
    3: "A3",
    4: "G1",
    5: "G2",
    6: "G3",
    7: "B1",
    8: "B2",
    9: "S1",
    10: "S2",
    11: "M",
    12: "L",
    13: "P",
}

# Full dictionary: distinguishes between dark (player=0) and light (player=1) pieces.
# For light pieces we assign codes in increasing order (lower code for higher-value pieces).
# For dark pieces, we use higher numbers (or reverse order) so that the more valuable pieces get a lower numerical code.
TILE_DICT_FULL = {
    0: {  # Dark pieces (player 0)
        "bQ": 0,   # Dark queen – lower numbers mean more valuable
        "bA1": 1,
        "bA2": 2,
        "bA3": 3,
        "bG1": 4,
        "bG2": 5,
        "bG3": 6,
        "bB1": 7,
        "bB2": 8,
        "bS1": 9,
        "bS2": 10,
        "bM": 11,
        "bL": 12,
        "bP": 13,
        "wP": 14,
        "wL": 15,
        "wM": 16,
        "wS1": 17,
        "wS2": 18,
        "wB1": 19,
        "wB2": 20,
        "wG1": 21,
        "wG2": 22,
        "wG3": 23,
        "wA1": 24,
        "wA2": 25,
        "wA3": 26,
        "wQ": 27,
        "EMPTY": 28,
    },
    1: {  # Light pieces (player 1) – order reversed relative to dark
        "wQ": 0,  # Light queen – highest value in the range 15-28
        "wA1": 1,
        "wA2": 2,
        "wA3": 3,
        "wG1": 4,
        "wG2": 5,
        "wG3": 6,
        "wB1": 7,
        "wB2": 8,
        "wS1": 9,
        "wS2": 10,
        "wM": 11,
        "wL": 12,
        "wP": 13,
        "bP": 14,
        "bL": 15,
        "bM": 16,
        "bS1": 17,
        "bS2": 18,
        "bB1": 19,
        "bB2": 20,
        "bG1": 21,
        "bG2": 22,
        "bG3": 23,
        "bA1": 24,
        "bA2": 25,
        "bA3": 26,
        "bQ": 27,
        "EMPTY": 28,
    },
}

INV_TILE_DICT_FULL = {
    0: {  # Dark pieces (player 0)
        0: "bQ",
        1: "bA1",
        2: "bA2",
        3: "bA3",
        4: "bG1",
        5: "bG2",
        6: "bG3",
        7: "bB1",
        8: "bB2",
        9: "bS1",
        10: "bS2",
        11: "bM",
        12: "bL",
        13: "bP",
        14: "wP",
        15: "wL",
        16: "wM",
        17: "wS1",
        18: "wS2",
        19: "wB1",
        20: "wB2",
        21: "wG1",
        22: "wG2",
        23: "wG3",
        24: "wA1",
        25: "wA2",
        26: "wA3",
        27: "wQ",
        28: "EMPTY",
    },
    1: {  # Light pieces (player 1) – order reversed relative to dark
        0: "wQ",
        1: "wA1",
        2: "wA2",
        3: "wA3",
        4: "wG1",
        5: "wG2",
        6: "wG3",
        7: "wB1",
        8: "wB2",
        9: "wS1",
        10: "wS2",
        11: "wM",
        12: "wL",
        13: "wP",
        14: "bP",
        15: "bL",
        16: "bM",
        17: "bS1",
        18: "bS2",
        19: "bB1",
        20: "bB2",
        21: "bG1",
        22: "bG2",
        23: "bG3",
        24: "bA1",
        25: "bA2",
        26: "bA3",
        27: "bQ",
        28: "EMPTY",
    },
}

DIRECTION_MAP_SUFFIX = {  # when indicator is at end (e.g., "wS1/")
    "/": 0,  # top right edge
    "-": 1,  # right-hand edge
    "\\": 2, # bottom right edge
}
DIRECTION_MAP_PREFIX = {  # when indicator is at beginning (e.g., "/wS1")
    "/": 3,  # bottom left edge
    "-": 4,  # left-hand edge
    "\\": 5, # top left edge
}

# Inverse mappings for direction indicators.
INV_DIRECTION_MAP_PREFIX = {
    0: "/",
    1: "-",
    2: "\\"
}

INV_DIRECTION_MAP_SUFFIX = {
    3: "/",
    4: "-",
    5: "\\"
}