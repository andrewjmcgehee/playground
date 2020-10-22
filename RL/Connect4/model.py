# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
  def __init__(self):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
    self.bn = nn.BatchNorm2d(128)

  def forward(self, x):
    x = x.view(-1, 3, 6, 7)
    x = F.relu(self.bn(self.conv(x)))
    return x

class ResBlock(nn.Module):
  def __init__(self):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(128)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
                           bias=False)
    self.bn2 = nn.BatchNorm2d(128)

  def forward(self, x):
    res = x
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)) + res)
    return out

class OutputBlock(nn.Module):
  def __init__(self):
    super(OutputBlock, self).__init__()
    # value output
    self.value_conv = nn.Conv2d(128, 3, kernel_size=1)
    self.value_bn = nn.BatchNorm2d(3)
    self.value_fc1 = nn.Linear(3 * 6 * 7, 32)
    self.value_fc2 = nn.Linear(32, 1)

    # policy output
    self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
    self.policy_bn = nn.BatchNorm2d(32)
    self.policy_lsm = nn.LogSoftmax(dim=1)
    self.policy_fc = nn.Linear(6 * 7 * 32, 7)

  def forward(self, x):
    # value
    v = F.relu(self.value_bn(self.value_conv(x)))
    v = v.view(-1, 3 * 6 * 7)
    v = F.relu(self.value_fc1(v))
    v = torch.tanh(self.value_fc2(v))
    # policy
    p = F.relu(self.policy_bn(self.policy_conv(x)))
    p = p.view(-1, 6 * 7 * 32)
    p = self.policy_fc(p)
    p = self.policy_lsm(p).exp()
    return v, p

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.conv = ConvBlock()
    self.res = [ResBlock() for _ in range(10)]
    self.out = OutputBlock()

  def forward(self, x):
    x = self.conv(x)
    for i in range(10):
      x = self.res[i](x)
    x = self.out(x)
    return x

class Loss(nn.Module):
  def __init__(self):
    super(Loss, self).__init__()

  def forward(self, y_value, value, y_policy, policy):
    value_loss = (value - y_value)**2
    policy_loss = torch.sum(
        (-policy * (1e-8 + y_policy.float()).float().log()), dim=1)
    total = (value_loss.view(-1).float() + policy_loss).mean()
    return total
