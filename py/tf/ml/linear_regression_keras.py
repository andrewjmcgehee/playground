# python
import logging

# third party
import tensorflow as tf

# custom
import constants
import util


def get_model():
  inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)
  x = inputs
  output = tf.keras.layers.Dense(1)(x)
  model = tf.keras.Model(inputs=inputs, outputs=output)
  logging.info(model.summary())
  return model


def train(x, y, model):
  opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  model.compile(optimizer=opt, loss='mse')
  model.fit(x, y, batch_size=32, epochs=10)
  logging.info(model.get_weights())
  util.plot_model_performance(constants._CUBIC_DATA, model)


def main():
  x, y = util.get_data(constants._CUBIC_DATA)
  model = get_model()
  train(x, y, model)


if __name__ == '__main__':
  main()