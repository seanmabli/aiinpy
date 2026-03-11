import numpy as np

class softmax:
  def forward(self, input):
    shifted = input - np.max(input)
    return shifted / np.sum(shifted)

  def backward(self, input):
    e = np.exp(input)
    s = np.sum(e)
    return (e * (s - e)) / (s * s)
