import numpy as np
from .tanh import tanh

class mish:
  def forward(self, input):
    return input * tanh.forward(np.log(1 + np.exp(input)))

  def backward(self, input):
    return (np.exp(input) * ((4 * np.exp(2 * input)) + np.exp(3 * input) + (4 * (1 + input)) + (np.exp(input) * (6 + (4 * input))))) / np.square(2 + (2 * np.exp(input)) + np.exp(2 * input))