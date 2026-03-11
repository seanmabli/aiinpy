import numpy as np

class prelu:
  def __init__(self, alpha):
    self.alpha = alpha

  def forward(self, input):
    return np.where(input >= 0, input, self.alpha * input)

  def backward(self, input):
    return np.where(input >= 0, 1.0, self.alpha)
