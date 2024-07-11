# python
import logging

logging.basicConfig(level=logging.INFO)

# third-party
import numpy as np

# custom
import constants
import util


def get_model(x, y, z):
  A = np.vstack([x, y, np.ones(len(x))]).T
  m1, m2, b = np.linalg.lstsq(A, z, rcond=None)[0]
  logging.info(f"{m1}, {m2}, {b}")
  return lambda x: m1 * x[:, 0] + m2 * x[:, 1] + b


def main():
  x, y, z = util.get_data(constants._NOISY_3D)
  model = get_model(x, y, z)
  util.plot_model_performance(constants._NOISY_3D, model)


if __name__ == '__main__':
  main()