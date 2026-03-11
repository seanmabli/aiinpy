import numpy as np

class selu:
  def forward(self, input):
    return 1.0507 * np.where(input < 0, 1.67326 * (np.exp(input) - 1), input)

  def backward(self, input):
    return 1.0507 * np.where(input < 0, 1.67326 * np.exp(input), 1.0)
