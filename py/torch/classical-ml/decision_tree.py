# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
import numpy as np
from sklearn import datasets

data = datasets.load_iris()


class Node:

  def __init__(self,
               feature=None,
               threshold=None,
               left=None,
               right=None,
               surprise=None,
               value=None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.surprise = surprise
    self.value = value


class DecisionTree:

  def __init__(self, min_split=2, max_depth=8):
    self.min_split = min_split
    self.max_depth = max_depth
    self.root = None
    self.num_features = None

  def entropy(self, x):
    counts = np.bincount(np.array(x, dtype=np.int64))
    proportions = counts / len(x)
    proportions = proportions[proportions != 0]
    result = (proportions * np.log2(proportions)).sum()
    return -result

  def surprise(self, parent, left, right):
    num_left = len(left) / len(parent)
    num_right = len(right) / len(parent)
    return (self.entropy(parent) - num_left * self.entropy(left) -
            num_right * self.entropy(right))

  def find_split(self, X, y):
    split = {}
    max_surprise = -1
    _, cols = X.shape
    parent = np.concatenate([X, y.reshape(-1, 1)], axis=-1)
    for c in range(cols):
      feature = X[:, c]
      for threshold in np.unique(feature):
        left = parent[parent[:, c] <= threshold, :]
        right = parent[parent[:, c] > threshold, :]
        if len(left) and len(right):
          y = parent[:, -1]
          y_left = left[:, -1]
          y_right = right[:, -1]
          surprise = self.surprise(y, y_left, y_right)
          if surprise > max_surprise:
            split = {
                "feature": c,
                "threshold": threshold,
                "left": left,
                "right": right,
                "surprise": surprise
            }
            max_surprise = surprise
    return split

  def build(self, X, y, depth=0):
    rows, _ = X.shape
    if rows >= self.min_split and depth <= self.max_depth:
      split = self.find_split(X, y)
      if split['surprise'] > 0:
        left = self.build(X=split['left'][:, :-1],
                          y=split['left'][:, -1],
                          depth=depth + 1)
        right = self.build(X=split['right'][:, :-1],
                           y=split['right'][:, -1],
                           depth=depth + 1)
        split.update({"left": left, "right": right})
        return Node(**split)
    y = y.astype(np.int64)
    return Node(value=np.bincount(y).argmax())

  def fit(self, X, y):
    self.num_features = X.shape[1]
    self.root = self.build(X, y)

  def _predict(self, x, node):
    if node.value is not None:
      return node.value
    value = x[node.feature]
    if value <= node.threshold:
      return self._predict(x, node=node.left)
    return self._predict(x, node=node.right)

  def predict(self, X):
    op = lambda r: self._predict(r, node=self.root)
    return np.apply_along_axis(op, 1, X).reshape(-1, 1)

  def _explain(self, node, depth, feature_names):
    if node is None:
      return
    if node.value is not None:
      print("  " * depth, end="")
      print(f"return {node.value}")
      return
    if node.left is not None and node.right is not None:
      print("  " * depth, end="")
      print(f"if {feature_names[node.feature]} <= {node.threshold}:")
      self._explain(node.left, depth + 1, feature_names)
      if node.right.value is not None:
        print("  " * depth + f"return {node.right.value}")
        return
      print("  " * depth + "else:")
      self._explain(node.right, depth + 1, feature_names)

  def explain(self, feature_names=None):
    """preorder traversal of the tree"""
    if self.root is None:
      print("model has not been fit to data yet")
      return
    if feature_names is None:
      feature_names = [f"x{i}" for i in range(self.num_features)]
    node = self.root
    self._explain(node, depth=0, feature_names=feature_names)