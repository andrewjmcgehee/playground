import numpy as np

def shifted_rastrigin(x, y):
  return ((x-5)**2 - 10 * np.cos(2 * np.pi * (x-5))) + ((y-5)**2 - 10 * np.cos(2 * np.pi * (y-5))) + 20
