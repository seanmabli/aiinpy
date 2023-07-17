import numpy as np

class sigmoid:
  def __repr__(self):
    return 'sigmoid()'

  def forward(self, input):
    return 1 / (1 + np.exp(-input))
    
  def backward(self, input):
    # return input * (1 - input) # this is when the input is already passed through the sigmoid function, the real derivative is below
    return np.exp(-input) / np.square(1 + np.exp(-input))