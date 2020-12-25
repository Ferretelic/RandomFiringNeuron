import torch

class RandomFiringNeuronFunction(torch.autograd.Function):
  @staticmethod
  def forward(self, x, weights, threshhold_proportion, forward_proportions, backward_proportions):
    threshhold = torch.randn(*x.shape, device="cpu") * threshhold_proportion
    activations = (x > threshhold)
    y = torch.empty_like(x)

    y[torch.logical_not(activations)] = x[torch.logical_not(activations)] * weights[0] + torch.randn(torch.sum(torch.logical_not(activations))) * forward_proportions[0]
    y[activations] = x[activations] * weights[1] + torch.randn(torch.sum(activations)) * forward_proportions[1]

    self.weights = weights
    self.backward_proportions = backward_proportions
    self.activations = activations

    return y

  @staticmethod
  def backward(self, dx):
    dy = torch.empty_like(dx)

    dy[torch.logical_not(self.activations)] = torch.abs(self.weights[0] + torch.randn(torch.sum(torch.logical_not(self.activations))) * self.backward_proportions[1]) * dx[torch.logical_not(self.activations)]
    dy[self.activations] = (self.weights[1] + torch.abs(torch.randn(torch.sum(self.activations))) * self.backward_proportions[1]) * dx[self.activations]

    return dy, None, None, None, None

class RandomFiringNeuron(torch.nn.Module):
  def __init__(self, weights=[0.1, 1], threshhold_proportion=0.5, forward_proportions=[0.2, 0.2], backward_proportions=[0.01, 0.01]):
    super(RandomFiringNeuron, self).__init__()
    self.weights = torch.nn.Parameter(torch.tensor(weights), requires_grad=False)
    self.threshhold_proportion = torch.nn.Parameter(torch.tensor(threshhold_proportion), requires_grad=False)
    self.forward_proportions = torch.nn.Parameter(torch.tensor(forward_proportions), requires_grad=False)
    self.backward_proportions = torch.nn.Parameter(torch.tensor(backward_proportions), requires_grad=False)

  def forward(self, input):
    return RandomFiringNeuronFunction.apply(input, self.weights, self.threshhold_proportion, self.forward_proportions, self.backward_proportions)
