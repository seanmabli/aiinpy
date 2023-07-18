import numpy as np

class stablesoftmax:
  def __repr__(self):
    return 'stablesoftmax()'

  def forward(self, input):
    a = np.exp(input - np.max(input))
    return a / np.sum(a)

  def newbackward(self, input): # use this
    forward = np.exp(input - np.max(input)) / np.sum(np.exp(input - np.max(input)))
    out = np.zeros((len(input), len(input)))
    for i in range(len(input)):
      for j in range(len(input)):
        out[i, j] = forward[i] * (1 if i == j else 0 - forward[j])
    return out

  def newbackwardtwo(self, input):
    input = input.flatten()
    s = np.exp(input - np.max(input)) / np.sum(np.exp(input - np.max(input)))
    print(s.shape)
    a = np.eye(s.shape[-1])
    temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
    temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
    temp1 = np.einsum('ij,jk->ijk',s,a)
    temp2 = np.einsum('ij,ik->ijk',s,s)
    return temp1-temp2

  def backward(self, input):
    return (np.exp(input) * (np.sum(np.exp(input)) - np.exp(input))) / np.sum(np.exp(input)) ** 2

# x = np.random.rand(10, 10)
# print(stablesoftmax().newbackwardtwo(x))