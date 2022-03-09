import numpy as np

class selu:
  def forward(self, input):
    equation = np.vectorize(self.equationforselu, otypes=[float])
    return 1.0507 * equation(input)

  def equationforselu(self, input):
    return 1.67326 * (np.exp(input) - 1) if (input < 0) else input

  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return 1.0507 * equation(input)

  def equationforderivative(self, input):
    return 1.67326 * np.exp(input) if (input < 0) else 1