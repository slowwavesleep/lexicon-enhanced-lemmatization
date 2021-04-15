"""
Constants for seq2seq models.
"""
from typing import List

PAD: str = "<PAD>"
PAD_ID: int = 0
UNK: str = "<UNK>"
UNK_ID: int = 1
SOS: str = "<SOS>"
SOS_ID: int = 2
EOS: str = "<EOS>"
EOS_ID: int = 3
FILL: str = "<FILL>"
FILL_ID: int = 4

VOCAB_PREFIX: List[str] = [PAD, UNK, SOS, EOS, FILL]

EMB_INIT_RANGE: float = 1.0
INFINITY_NUMBER: int = 1e12
