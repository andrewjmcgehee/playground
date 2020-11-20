# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.nn import BatchNorm2d, Conv2d, Linear, LogSoftmax, Module
from torch.nn.functional import relu

RES_TOWER = 9

class ConvBlock(Module):
  '''
  Simple convolution block. Takes 3 in channels and puts out 128 output channels.
  We are using 3x3 kernels with stride and padding of 1, so after each
  convolution our input tensor doesn't grow or shrink. We also apply a simple
  batch norm layer also with 128 output channels.

    In Shape:  (B, I, R, C)
      B - Batch size
      I - Number of input channels (in our case 3 for the red binary matrix, the
          yellow binary matrix, and the turn binary matrix we discussed)
      R - Number of rows
      C - Number of columns
    Out Shape: (B, O, R, C)
      B - Batch size
      O - Output channels (in our case 128, which is half the reported value in
          the AlphaZero paper)
      R - Number of rows
      C - Number of columns
  '''
  def __init__(self):
    super(ConvBlock, self).__init__()
    self.conv = Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
    self.bn = BatchNorm2d(128)

  def forward(self, x):
    # resize input dimensions to fit (B, I, R, C)
    x = x.view(-1, 3, 6, 7)
    return relu(self.bn(self.conv(x)))

class ResBlock(Module):
  '''
  Simple residual block as defined in the original residual neural net paper.
  The input first goes through a convolution weight layer, a batch norm layer,
  and a ReLU activation. Then its passed through a second convolution weight
  layer and batch norm layer. Then the original input follows the skip path and
  is added prior to the final ReLU activation.

    In Shape: (B, I, R, C)
      B - Batch size
      I - Number of input channels (128 from previous convolution layer)
      R - Number of rows
      C - Number of columns
    Out Shape: (B, O, R, C)
      B - Batch size
      O - Output channels (again 128 to be able to use a residual 'tower')
      R - Number of rows
      C - Number of columns
  '''
  def __init__(self):
    super(ResBlock, self).__init__()
    self.conv1 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
                        bias=False)
    self.bn1 = BatchNorm2d(128)
    self.conv2 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
                        bias=False)
    self.bn2 = BatchNorm2d(128)

  def forward(self, x):
    res = x
    out = relu(self.bn1(self.conv1(x)))
    return relu(self.bn2(self.conv2(out)) + res)

class OutputBlock(Module):
  '''
  Output block for the value head and the policy head. The value head outputs
  a single scalar. The policy head outputs a valid probability mass function
  representing the relative favorability of each action. Both are combined into
  a single
  '''
  def __init__(self):
    super(OutputBlock, self).__init__()
    '''
    I still don't understand the intuition of the initial convolution layer
    here, but it is reflected in the AlphaGo Zero paper as a convolution of
    one 1x1 filter followed by batch norm and ReLU. The rest is straight forward
    enough, but we modified the hidden layer size (down to 32 from 256) to fit
    our use case.
    '''
    # value output
    self.value_conv = Conv2d(128, 1, kernel_size=1)
    self.value_bn = BatchNorm2d(1)
    self.value_fc1 = Linear(6 * 7, 32)
    self.value_fc2 = Linear(32, 1)

    '''
    Similar issue here where I don't catch the intuition, but we're following
    the AlphaGo Zero arhitecture (since AlphaZero itself isn't fully public).
    '''
    # policy output
    self.policy_conv = Conv2d(128, 2, kernel_size=1)
    self.policy_bn = BatchNorm2d(2)
    self.policy_lsm = LogSoftmax(dim=1)
    self.policy_fc = Linear(6 * 7 * 2, 7)

  def forward(self, x):
    # value: in -> conv -> batch norm -> ReLU -> dense -> tanh
    v = relu(self.value_bn(self.value_conv(x)))
    v = v.view(-1, 6 * 7)
    v = relu(self.value_fc1(v))
    v = torch.tanh(self.value_fc2(v))
    # policy: in -> conv -> batch norm -> ReLU -> dense -> log softmax
    p = relu(self.policy_bn(self.policy_conv(x)))
    p = p.view(-1, 6 * 7 * 2)
    p = self.policy_fc(p)
    p = self.policy_lsm(p).exp()
    return v, p

class Model(Module):
  '''
  Putting it altogether in a smaller version of the AlphaGo Zero model
  '''
  def __init__(self):
    super(Model, self).__init__()
    self.conv = ConvBlock()
    for i in range(RES_TOWER):
      setattr(self, f'res_{i}', ResBlock())
    self.out = OutputBlock()

  def forward(self, x):
    x = self.conv(x)
    for i in range(RES_TOWER):
      x = getattr(self, f'res_{i}')(x)
    return self.out(x)

class Loss(Module):
  '''
  Simple mean squared error function that handles our value and policy head
  together.
  '''
  def __init__(self):
    super(Loss, self).__init__()

  def forward(self, y_value, value, y_policy, policy):
    value_loss = (value - y_value)**2
    policy_loss = torch.sum(
        (-policy * (1e-8 + y_policy.float()).float().log()), dim=1)
    total = (value_loss.view(-1).float() + policy_loss).mean()
    return total
