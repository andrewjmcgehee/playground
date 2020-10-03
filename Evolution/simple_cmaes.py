import numpy as np

def shifted_rastrigin(x, y, shift=5):
  return (
    ( (x-shift)**2 - 10 * np.cos(2 * np.pi * (x-shift)) ) +
    ( (y-shift)**2 - 10 * np.cos(2 * np.pi * (y-shift)) ) + 20
  )
