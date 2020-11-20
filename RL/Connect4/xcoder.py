import numpy as np
from board import Board, NUM_COLS, NUM_ROWS, RED, YELLOW

'''
Encodes a board to a one-hot encoding where the 0th axis maps to red pieces,
the 1st axis maps to yellow pieces, and the 2nd axis maps to whose turn it is.
Returns a NP array of size (ROWS, COLS, 3)
'''
def encode(b):
  state = b.current_board
  encoded = np.zeros(shape=(NUM_ROWS, NUM_COLS, 3), dtype=np.uint8)
  red_mask = np.where(state == RED)
  yellow_mask = np.where(state == YELLOW)
  encoded[:,:,0][red_mask] = 1
  encoded[:,:,1][yellow_mask] = 1
  if b.player == RED:
    encoded[:,:,2] = 1
  return encoded

'''
Decodes an encoded board. Returns a Board() object
'''
def decode(encoded):
  decoded = np.zeros(shape=(NUM_ROWS, NUM_COLS), dtype=np.uint8)
  red_mask = np.where(encoded[:,:,0] == 1)
  yellow_mask = np.where(encoded[:,:,1] == 1)
  decoded[red_mask] = RED
  decoded[yellow_mask] = YELLOW
  b = Board()
  b.current_board = decoded
  if encoded[0,0,2] == RED
    b.player = RED
  else:
    b.player = YELLOW
  return b
