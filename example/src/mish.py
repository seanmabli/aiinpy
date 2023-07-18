import numpy as np

class mish:
  def __repr__(self):
    return 'mish()'

  def forward(self, input):
    return (input * ((2 * np.exp(input)) + np.exp(2 * input))) / ((2 * np.exp(input)) + np.exp(2 * input) + 2)

  def backward(self, input):
    return (np.exp(input) * ((4 * np.exp(2 * input)) + np.exp(3 * input) + (4 * (1 + input)) + (np.exp(input) * (6 + (4 * input))))) / np.square(2 + (2 * np.exp(input)) + np.exp(2 * input))