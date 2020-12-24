import numpy as np
import torch

class RandomFiringNeuron(torch.autograd.Function):
  def __init__(self, proportions=[0.5, 0.2], weights=[0.1, 1], gradient_proportions=[0.01, 0.01]):
    self.proportions = proportions
    self.weights = weights
    self.gradient_proportions = gradient_proportions

  @staticmethod
  def forward(self, x):
    threshhold = np.random.randn(x.shape[0]) * self.proportions[0]
    self.activation = np.where(x <= threshhold, False, True)
    y = np.empty_like(x)

    y[np.logical_not(self.activation)] = x[np.logical_not(self.activation)] * self.weights[0] + np.random.randn(np.sum(np.logical_not(self.activation))) * self.proportions[1]
    y[self.activation] = x[self.activation] * self.weights[1] + np.random.randn(np.sum(self.activation)) * self.proportions[1]

    return y

  @staticmethod
  def backward(self, d_input):
    d_output = np.empty_like(d_input)
    d_output[np.logical_not(self.activation)] = self.weights[0] + np.random.randn(np.sum(np.logical_not(self.activation))) * self.gradient_proportions[0]
    d_output[self.activation] = self.weights[1] + np.random.randn(np.sum(self.activation)) * self.gradient_proportions[1]

    return d_output
