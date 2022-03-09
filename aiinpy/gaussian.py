import numpy as np

class gaussian:
  def forward(self, input):
    return np.exp(-np.square(input))
    
  def backward(self, input):
    return -2 * input * np.exp(-np.square(input))