import numpy as np

class softmax:
  def __repr__(self):
    return 'softmax()'

  def forward(self, input):
    return (input - np.max(input)) / np.sum(input - np.max(input))

  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return equation(input, np.sum(np.exp(input)))

  def equationforderivative(self, input, SumExpOfinput):
    return (np.exp(input) * (SumExpOfinput - np.exp(input))) / (SumExpOfinput) ** 2