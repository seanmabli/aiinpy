import numpy as np

class tanh:
  def forward(self, input):
    return np.tanh(input)

  def backward(self, input):
    return 1 - np.square(input)