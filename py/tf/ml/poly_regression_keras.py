# python
import logging

logging.basicConfig(level=logging.INFO)

# third-party
import tensorflow as tf

# custom
import constants
import util


def get_model(hidden_sizes):
  inputs = tf.keras.Input((1,), dtype=tf.float32)
  x = inputs
  for i, size in enumerate(hidden_sizes):
    x = tf.keras.layers.Dense(size, activation='relu', name=f'hidden_{i}')(x)
  output = tf.keras.layers.Dense(1)(x)
  model = tf.keras.Model(inputs=inputs, outputs=output)
  logging.info(model.summary())
  return model


def train(x, y, model):
  opt = tf.keras.optimizers.Adam(0.01)
  model.compile(optimizer=opt, loss='mse')
  model.fit(x, y, batch_size=32, epochs=10)


def main():
  x, y = util.get_data(constants._CUBIC_DATA)
  model = get_model([80, 30, 10])
  train(x, y, model)
  util.plot_model_performance(constants._CUBIC_DATA, model)


if __name__ == '__main__':
  main()