import numpy as np

class softmax:
  def __repr__(self):
    return 'softmax()'

  def forward(self, input):
    return np.exp(input) / np.sum(np.exp(input))

  def backward(self, input):
    forward = np.exp(input - np.max(input)) / np.sum(np.exp(input - np.max(input)))
    out = np.zeros((len(input), len(input)))
    for i in range(len(input)):
      for j in range(len(input)):
        out[i, j] = forward[i] * (1 if i == j else 0 - forward[j])
    return out