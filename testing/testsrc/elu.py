import numpy as np

class elu:
  def __init__(self, alpha):
    self.alpha = alpha
    self.forwardequation = np.vectorize(self.equationforelu, otypes=[float])
    self.backwardequation = np.vectorize(self.equationforderivative, otypes=[float])

  def forward(self, input):
    return self.forwardequation(input)

  def equationforelu(self, input):
    return self.alpha * (np.exp(input) - 1) if input <= 0 else input

  def backward(self, input):
    return self.backwardequation(input)

  def equationforderivative(self, input):
    if input < 0:
      return self.alpha * np.exp(input)
    elif input > 0:
      return 1
    elif input == 0 and self.alpha == 1:
      return 1
