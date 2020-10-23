import numpy as np
import board

'''
Encodes a board to a one-hot encoding where the 0th axis maps to red pieces,
the 1st axis maps to yellow pieces, and the 2nd axis maps to whose turn it is.
Returns a NP array of size (ROWS, COLS, 3)
'''
def encode(b):
  state = b._current_board
  encoded = np.zeros(shape=(board.NUM_ROWS, board.NUM_COLS, 3)).astype('uint8')
  red_mask = np.where(state == board.RED)
  yellow_mask = np.where(state == board.YELLOW)
  encoded[:,:,0][red_mask] = 1
  encoded[:,:,1][yellow_mask] = 1
  if b.player == board.RED:
    encoded[:,:,2] = 1
  return encoded

'''
Decodes an encoded board. Returns a Board() object
'''
def decode(encoded):
  decoded = np.zeros(shape=(board.NUM_ROWS, board.NUM_COLS)).astype('uint8')
  red_mask = np.where(encoded[:,:,0] == 1)
  yellow_mask = np.where(encoded[:,:,1] == 1)
  decoded[red_mask] = board.RED
  decoded[yellow_mask] = board.YELLOW
  b = board.Board()
  b._current_board = decoded
  if (encoded[:,:,2] == 1).all():
    b.player = board.RED
  else:
    b.player = board.YELLOW
  return b
