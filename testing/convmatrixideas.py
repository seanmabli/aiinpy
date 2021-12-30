import numpy as np
import testsrc as ai
import time

class convmatrix:
  def __init__(self, weight, inshape, outshape):
    self.weight, self.outshape = weight, outshape
    
    self.filter = []
    self.flattenweight = np.array(self.weight[0])
    for row in self.weight[1:]:
      self.flattenweight = np.append(self.flattenweight, np.zeros(inshape[0] - self.weight.shape[0]))
      self.flattenweight = np.append(self.flattenweight, row)

    i = 0
    while i + len(self.flattenweight) <= np.prod(inshape):
      y = np.zeros((np.prod(inshape)))
      y[i : i + len(self.flattenweight)] = self.flattenweight
      self.filter.append(y)
      if (np.prod(inshape) - (i + len(self.flattenweight))) % inshape[0] == 0:
        i += self.weight.shape[0]
      else:
        i += 1
    self.filter = np.array(self.filter)

  def forward(self, input):
    self.input = input
    self.out = self.filter @ input.flatten()
    return self.out # self.out.reshape(self.outshape)

  def backward(self, outerror):
    self.outGradient = outerror * self.out # outerror.flatten() * self.out
    self.FilterΔ = np.outer(self.input.flatten().T, self.outGradient).reshape(self.filter.shape)
    return np.flip((self.filter.T @ outerror).reshape(self.input.shape), axis=1) # np.flip(self.filter.T.dot(outerror.flatten()).reshape(self.input.shape), axis=1)

input = np.zeros((28, 28))
weightshape = (3, 3)

conv = ai.conv(inshape=input.shape, filtershape=weightshape, learningrate=0)
start = time.time()
convout = conv.forward(input)
convinerror = conv.backward(1 - convout)
q = time.time() - start

convmatrix = convmatrix(weight=conv.Filter.reshape(weightshape), inshape=input.shape, outshape=convout.shape)
start = time.time()
convmatrixout = convmatrix.forward(input)
convmatrixinerror = convmatrix.backward(1 - convmatrixout)
w = time.time() - start

print(q/w)