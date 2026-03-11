import numpy as np

class stablesoftmax:
  def forward(self, input):
    e = np.exp(input - np.max(input))
    return e / np.sum(e)

  def backward(self, input):
    e = np.exp(input)
    s = np.sum(e)
    return (e * (s - e)) / (s * s)
