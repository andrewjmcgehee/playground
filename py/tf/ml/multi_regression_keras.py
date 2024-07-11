# python
import logging

logging.basicConfig(level=logging.INFO)

# third-party
import numpy as np
import tensorflow as tf

# custom
import constants
import util


def get_model():
  inputs = tf.keras.Input((2,), dtype=tf.float32)
  output = tf.keras.layers.Dense(1)(inputs)
  model = tf.keras.Model(inputs=inputs, outputs=output)
  logging.info(model.summary())
  return model


def train(x, y, model):
  opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  model.compile(optimizer=opt, loss='mse')
  model.fit(x, y, batch_size=32, epochs=10)
  logging.info(model.weights)
  util.plot_model_performance(constants._NOISY_3D, model)


def main():
  x, y, z = util.get_data(constants._NOISY_3D)
  feature, label = np.vstack([x, y]).T, z
  model = get_model()
  train(feature, label, model)


if __name__ == '__main__':
  main()