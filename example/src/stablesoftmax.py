import numpy as np

class stablesoftmax:
  def __repr__(self):
    return 'stablesoftmax()'

  def forward(self, input):
    return np.exp(input - np.max(input)) / np.sum(np.exp(input - np.max(input)))

  def backward(self, input):
    forward = np.exp(input - np.max(input)) / np.sum(np.exp(input - np.max(input)))
    out = np.zeros((len(input), len(input)))
    for i in range(len(input)):
      for j in range(len(input)):
        out[i, j] = input[i] * (1 - input[j]) if i == j else - input[i] * input[j]
    return np.sum(out, axis=0)

x = [1, 5, 7]
z = stablesoftmax().forward(x)
y = stablesoftmax().backward(x)
print(np.round(z, 3))
print(np.round(y, 3))