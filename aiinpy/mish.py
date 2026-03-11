import numpy as np

class mish:
  def forward(self, input):
    return input * np.tanh(np.log(1 + np.exp(input)))

  def backward(self, input):
    e = np.exp(input)
    e2 = e * e
    e3 = e2 * e
    d = 2 + 2 * e + e2
    return (e * (4 * e2 + e3 + 4 * (1 + input) + e * (6 + 4 * input))) / (d * d)
