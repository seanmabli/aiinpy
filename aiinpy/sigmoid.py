import numpy as np

class sigmoid:
  def forward(self, input):
    return 1 / (1 + np.exp(-input))
    
  def backward(self, input):
    return input * (1 - input)
