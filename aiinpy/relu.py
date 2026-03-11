import numpy as np

class relu:
  def forward(self, input):
    return np.maximum(0, input)

  def backward(self, input):
    return np.where(input > 0, 1.0, 0.0)
