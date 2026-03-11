import numpy as np

class silu:
  def forward(self, input):
    s = 1.0 / (1 + np.exp(-input))
    return input * s

  def backward(self, input):
    s = 1.0 / (1 + np.exp(-input))
    return s * (1 + input * (1 - s))
