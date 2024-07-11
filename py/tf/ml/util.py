from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import constants

_SCATTER_LINE_DATA = (constants._CUBIC_DATA, constants._CUBIC_DATA_TIGHT)
_3D_DATA = (constants._NOISY_3D)


def get_data(filename):
  df = pd.read_csv(filename)
  if filename in _SCATTER_LINE_DATA:
    return df['features'], df['label']
  if filename in _3D_DATA:
    return df['x'], df['y'], df['z']
  raise ValueError(f'unrecognized filename: {filename}')


def plot_model_performance(data_file, model):
  if data_file in _SCATTER_LINE_DATA:
    x, y = get_data(data_file)
    generating_fn = lambda x: x**3 / 400
    _scatter_line_plot(x, y, model, generating_fn=generating_fn)
  elif data_file in _3D_DATA:
    x, y, z = get_data(data_file)
    generating_fn = lambda x, y: 3*x + 4*y
    _scatter_plane_plot(x, y, z, model, generating_fn=generating_fn)


def _scatter_line_plot(x, y, model, generating_fn=None):
  x, y = map(np.array, zip(*sorted(zip(x, y))))
  plt.scatter(x, y, label='true', alpha=0.2, color='C0')
  plt.plot(x, model(x), label='pred', color='C1')
  if generating_fn is not None:
    plt.plot(x, generating_fn(x), label='generator', color='black')
  plt.legend()
  plt.tight_layout()
  plt.show()


def _scatter_plane_plot(x, y, z, model, generating_fn=None):
  N = len(x)
  side_length = int(np.sqrt(N))
  z_pred = model(np.vstack([x, y]).T)
  if not isinstance(z_pred, np.ndarray):
    z_pred = z_pred.numpy()
  z_pred = z_pred.reshape((side_length, side_length))
  x = x.to_numpy().reshape((side_length, side_length))
  y = y.to_numpy().reshape((side_length, side_length))
  z = z.to_numpy().reshape((side_length, side_length))
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(x, y, z, label='true', alpha=0.2, color='C0')
  ax.plot_surface(x, y, z_pred, label='pred', color='C1')
  if generating_fn is not None:
    ax.plot_surface(x,
                    y,
                    generating_fn(x, y),
                    label='generator',
                    color='black',
                    alpha=0.2)
  plt.show()