import numpy as np

class leakyrelu:
  def __init__(self, alpha=0.01):
    self.alpha = alpha

  def forward(self, input):
    return np.maximum(self.alpha * input, input)

  def backward(self, input):
    return np.where(input >= 0, 1.0, self.alpha)
