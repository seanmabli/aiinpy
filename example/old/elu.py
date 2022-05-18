import numpy as np

class elu:
  def __init__(self, alpha):
    self.alpha = alpha
    self.forwardequationvectorized = np.vectorize(self.forwardequation, otypes=[float])
    self.backwardequationvectorized = np.vectorize(self.backwardequation, otypes=[float])

  def forward(self, input):
    return self.forwardequationvectorized(input)

  def forwardequation(self, input):
    return self.alpha * (np.exp(input) - 1) if input <= 0 else input

  def backward(self, input):
    return self.backwardequationvectorized(input)

  def backwardequation(self, input):
    if input < 0:
      return self.alpha * np.exp(input)
    elif input > 0:
      return 1
    elif input == 0 and self.alpha == 1:
      return 1
