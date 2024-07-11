# third-party
import numpy as np

# custom
import constants
import util


def get_model(x, y):
  A = np.vstack([x, np.ones(len(x))]).T
  m, b = np.linalg.lstsq(A, y, rcond=None)[0]
  return lambda x: m*x + b


def main():
  x, y = util.get_data(constants._CUBIC_DATA)
  model = get_model(x, y)
  util.plot_model_performance(constants._CUBIC_DATA, model)


if __name__ == "__main__":
  main()