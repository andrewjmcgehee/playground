from collections import defaultdict
from copy import deepcopy
import math

import numpy as np
import torch

from board import RED
from xcoder import encode

C_PUCT = 4
DIRICHLET_P = 0.25
DIRICHLET_A = 1.75

class Root:
  '''
  Dummy class for the parent of the root of our tree. We just need access to
  the children_V and children_N of the root's parent. So we need this class.
  '''
  def __init__(self):
    self.parent = None
    self.children_V = defaultdict(float)
    self.children_N = defaultdict(int)

class UCTNode:
  '''
  Upper confidence tree node. Tracks the priors P(s, a) of the children, the
  summed value of the children's subtree (children_V) and the number of times
  the child has been visited (children_N).
  '''
  def __init__(self, board, action, parent=None):
    self.board = board
    self.action = action
    self.expanded = False
    self.parent = parent
    self.children = dict()

    self.children_priors = np.zeros(shape=7, dtype=np.float32)
    self.children_V = np.zeros(shape=7, dtype=np.float32)
    self.children_N = np.zeros(shape=7, dtype=np.int32)

  '''
  Property getter / setters for this nodes V (summed value of subtree) and N
  (number of times visited).
  '''
  @property
  def N(self):
    return self.parent.children_N[self.action]

  @N.setter
  def N(self, value):
    self.parent.children_N[self.action] = value

  @property
  def V(self):
    return self.parent.children_V[self.action]

  @V.setter
  def V(self, value):
    self.parent.children_V[self.action] = value

  '''
  The Q (quality) of the children is the children_V / children_N. We add 1 to
  children_N to prevent division by zero.
  '''
  def children_Q(self):
    return self.children_V / (1 + self.children_N)

  '''
  This is the confidence term for each of the children. Calculated by the
  typical PUCT formula with hyper-parameter C_PUCT.
  '''
  def children_U(self):
    return C_PUCT * (
        math.sqrt(self.N) * abs(self.children_priors) / (1 + self.children_N))

  '''
  Allows us to imagine what state "board" would be in if we took "action."
  '''
  def imagine_state(self, board, action):
    board.place_piece(action)
    return board

  '''
  Gets the best child to explore as defined by our quality and upper confidence.
  '''
  def best_child(self):
    move_quality = self.children_Q() + self.children_U()
    if self.action_indices:
      index = np.argmax(move_quality[self.action_indices])
      return self.action_indices[index]
    return np.argmax(move_quality)

  '''
  Expand a node based on the legal actions available. Applies dirichlet noise
  if the node is the root.
  '''
  def expand(self, priors):
    self.expanded = True
    self.action_indices = self.board.actions()
    if not self.action_indices:
      self.expanded = False
    # create mask for illegal actions
    illegal_mask = np.ones(priors.shape, dtype=bool)
    illegal_mask[self.action_indices] = False
    # probability of illegal actions needs to be zero
    priors[illegal_mask] = 0
    if self.parent is None:
      priors = self.apply_dirichlet_noise(self.action_indices, priors)
    self.children_priors = priors

  '''
  Applies dirichlet noise to the valid actions in "priors" using hyperparameters
  DIRICHLET_A (the array passed into the Dir(a) function) and DIRICHLET_P (the
  weighting of the noise).
  '''
  def apply_dirichlet_noise(self, action_indices, priors):
    alpha = 1 - DIRICHLET_P
    beta = DIRICHLET_P
    valid_priors = priors[action_indices]
    valid_priors *= alpha
    noise = beta * np.random.dirichlet(
        np.zeros(shape=len(valid_priors), dtype=np.float32) + DIRICHLET_A)
    priors[action_indices] = valid_priors + noise
    return priors

  '''
  Selects a leaf node in the Monte Carlo tree (not the game tree).
  '''
  def select_leaf(self):
    current = self
    while current.expanded:
      best_action = current.best_child()
      current = current.cached_child(best_action)
    return current

  '''
  Creates a node for a state if it doesn't exist along a path of exploration.
  Note that it does not expand this node.
  '''
  def cached_child(self, action):
    if action not in self.children:
      board = deepcopy(self.board)
      board = self.imagine_state(board, action)
      self.children[action] = UCTNode(board, action, parent=self)
    return self.children[action]

  '''
  Propagates a value from a completed rollout up the monte carlo tree.
  '''
  def backtrack(self, value_estimate):
    current = self
    while current.parent is not None:
      current.N += 1
      mult = 1 if current.board.player == RED else -1
      current.V += mult * value_estimate
      current = current.parent

'''
The actual Monte Carlo Tree Search method.
'''
def mcts(board, model, iters=500):
  root = UCTNode(board, action=None, parent=Root())
  for i in range(iters):
    leaf = root.select_leaf()
    encoded_state = encode(leaf.board).transpose(2, 0, 1)
    encoded_state = torch.from_numpy(encoded_state).float()
    value, policy = model(encoded_state)
    value = value.item()
    policy = policy.detach().cpu().numpy().reshape(-1)
    if leaf.board.winner() or not leaf.board.actions():
      leaf.backtrack(value)
      continue
    leaf.expand(policy)
    leaf.backtrack(value)
  return root
