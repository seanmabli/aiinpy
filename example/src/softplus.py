import numpy as np

class softplus:
  def __repr__(self):
    return 'softplus()'

  def forward(self, input):
    return np.log(1 + np.exp(input))
    
  def backward(self, input):
    return 1 / (1 + np.exp(-input))