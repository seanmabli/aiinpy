import numpy as np

class prelu:
  def __init__(self, alpha):
    self.alpha = alpha
    self.forwardequationvectorized = np.vectorize(self.forwardequation, otypes=[float])
    self.backwardequationvectorized = np.vectorize(self.backwardequation, otypes=[float])

  def forward(self, input):
    return self.forwardequationvectorized(input)

  def forwardequation(self, input):
    return input if input >= 0 else self.alpha * input

  def backward(self, input):
    return self.backwardequationvectorized(input)

  def backwardequation(self, input):
    return 1 if input >= 0 else self.alpha
