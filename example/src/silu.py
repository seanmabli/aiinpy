import numpy as np

class silu:
  def forward(self, input):
    return input / (1 + np.exp(-input))
    
  def backward(self, input):
    return (1 + np.exp(-input) + (input * np.exp(-input))) / np.square(1 + np.exp(-input))