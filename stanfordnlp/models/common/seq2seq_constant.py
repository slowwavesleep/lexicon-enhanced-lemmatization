"""
Constants for seq2seq models.
"""

PAD = '<PAD>'
PAD_ID = 0
UNK = '<UNK>'
UNK_ID = 1
SOS = '<SOS>'
SOS_ID = 2
EOS = '<EOS>'
EOS_ID = 3
FILL = '<FILL>'
FILL_ID = 4

VOCAB_PREFIX = [PAD, UNK, SOS, EOS, FILL]

EMB_INIT_RANGE = 1.0
INFINITY_NUMBER = 1e12
