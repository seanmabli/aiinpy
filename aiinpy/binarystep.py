import numpy as np

class binarystep:
  def forward(self, input):
    return np.where(input >= 0, 1.0, 0.0)

  def backward(self, input):
    return 1
