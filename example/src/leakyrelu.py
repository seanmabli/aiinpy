import numpy as np

class leakyrelu:
  def __init__(self, alpha=0.01):
    self.alpha = alpha

  def forward(self, input):
    return np.maximum(self.alpha * input, input)

  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return equation(input)

  def equationforderivative(self, input):
    return self.alpha if input < 0 else 1