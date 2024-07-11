"""Linear regression models."""
import numpy as np
from sklearn import datasets
from sklearn import model_selection
import torch
from torch import nn
from torch import optim

_MULTIPLE = "multiple"
_MULTIVAR = "multivar"
_POLY = "poly"
_SIMPLE = "simple"


class LinearRegressor(nn.Module):

  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size, dtype=torch.float32)

  def forward(self, x):
    return self.linear(x)


class PolynomialRegressor(nn.Module):

  def __init__(self, degree):
    super().__init__()
    power = torch.arange(degree + 1, requires_grad=False)
    self.register_buffer('power', power)
    self.linear = nn.Linear(degree + 1, 1, bias=False)

  def forward(self, x):
    x = torch.pow(x, self.power)
    x = self.linear(x)
    return x


class IrisData(torch.utils.data.Dataset):

  def __init__(self, x, y):
    super().__init__()
    self.x = x
    self.y = y

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


class EarlyStopper:

  def __init__(self, patience=50, min_delta=1e-6):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = np.inf

  def early_stop(self, validation_loss):
    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False


def get_train_test_loaders(model_type, return_raw_data=False):
  data, _ = datasets.load_iris(return_X_y=True)
  sepal_L = data[:, 0].reshape(-1, 1).astype(np.float32)
  sepal_W = data[:, 1].reshape(-1, 1).astype(np.float32)
  petal_L = data[:, 2].reshape(-1, 1).astype(np.float32)
  petal_W = data[:, 3].reshape(-1, 1).astype(np.float32)
  if model_type == _SIMPLE:
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        sepal_L, petal_L, test_size=0.2, random_state=1)
  elif model_type == _MULTIPLE:
    sepal = np.concatenate([sepal_L, sepal_W], axis=-1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        sepal, petal_L, test_size=0.2, random_state=1)
  elif model_type == _MULTIVAR:
    sepal = np.concatenate([sepal_L, sepal_W], axis=-1)
    petal = np.concatenate([petal_L, petal_W], axis=-1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        sepal, petal, test_size=0.2, random_state=1)
  elif model_type == _POLY:
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        sepal_L, petal_L, test_size=0.2, random_state=1)
  else:
    raise ValueError(f"model_type not supported {model_type}")
  train_data = IrisData(x_train, y_train)
  test_data = IrisData(x_test, y_test)
  train_load = torch.utils.data.DataLoader(train_data,
                                           shuffle=True,
                                           batch_size=2)
  test_load = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=2)
  if return_raw_data:
    return x_train, y_train, x_test, y_test
  return train_load, test_load


def train_epoch(train_load, model, optimizer, loss_fn, device):
  model.train()
  running_loss = 0.0
  for _, (inputs, labels) in enumerate(train_load, 0):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  return running_loss / len(train_load)


def val_epoch(test_load, model, loss_fn, device):
  model.eval()
  running_loss = 0.0
  for _, (inputs, labels) in enumerate(test_load, 0):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    running_loss += loss.item()
  return running_loss / len(test_load)


def _train(epochs, train_load, test_load, model):
  device = get_device()
  model = model.to(device)
  optimizer = optim.Adam(model.parameters())
  loss_fn = nn.MSELoss()
  stopper = EarlyStopper()
  for epoch in range(epochs):
    train_loss = train_epoch(train_load, model, optimizer, loss_fn, device)
    val_loss = val_epoch(test_load, model, loss_fn, device)
    if epoch == 0 or epoch % int(np.sqrt(epochs)) == 0:
      print(f"epoch {epoch+1:04d}/{epochs} train loss: {train_loss}")
      print(f"                val loss: {val_loss}")
    if stopper.early_stop(val_loss):
      break
  return model


def get_device():
  device = torch.device("cpu")
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  return device


def train_simple_regression(epochs):
  train_load, test_load = get_train_test_loaders(model_type=_SIMPLE)
  model = LinearRegressor(input_size=1, output_size=1)
  model = _train(epochs, train_load, test_load, model)
  return model


def train_multiple_regression(epochs):
  train_load, test_load = get_train_test_loaders(model_type=_MULTIPLE)
  model = LinearRegressor(input_size=2, output_size=1)
  model = _train(epochs, train_load, test_load, model)
  return model


def train_multivariate_regression(epochs):
  train_load, test_load = get_train_test_loaders(model_type=_MULTIVAR)
  model = LinearRegressor(input_size=2, output_size=2)
  model = _train(epochs, train_load, test_load, model)
  return model


def train_polynomial_regression(epochs):
  train_load, test_load = get_train_test_loaders(model_type=_POLY)
  model = PolynomialRegressor(degree=3)
  model = _train(epochs, train_load, test_load, model)
  return model
