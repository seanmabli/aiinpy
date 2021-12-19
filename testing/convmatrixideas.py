import numpy as np
import testsrc as ai

class convmatrix:
  def __init__(self, weight, outshape):
    self.weight, self.outshape = weight, outshape

  def forward(self, input):
    self.input = input
    self.filter = []
    self.flattenweight = np.array(self.weight[0])
    for row in self.weight[1:]:
      self.flattenweight = np.append(self.flattenweight, np.zeros(input.shape[0] - self.weight.shape[0]))
      self.flattenweight = np.append(self.flattenweight, row)

    i = 0
    while i + len(self.flattenweight) <= np.prod(input.shape):
      y = np.zeros((np.prod(input.shape)))
      y[i : i + len(self.flattenweight)] = self.flattenweight
      self.filter.append(y)
      if (np.prod(input.shape) - (i + len(self.flattenweight))) % input.shape[0] == 0:
        i += self.weight.shape[0]
      else:
        i += 1
    self.filter = np.array(self.filter)
    self.out = self.filter.dot(input.flatten()).reshape(self.outshape)
    return self.out

  def backward(self, outerror):
    self.outGradient = outerror.flatten() * self.out.flatten()
    self.FilterΔ = np.outer(self.input.flatten().T, self.outGradient).reshape(self.filter.shape)
    return np.flip(self.filter.T.dot(outerror.flatten()).reshape(self.input.shape), axis=1)

input = np.zeros((10, 10))
weightshape = (9, 9)

conv = ai.conv(inshape=input.shape, filtershape=weightshape, learningrate=0)
convout = conv.forward(input)
convinerror = conv.backward(1 - convout)
convdelta = conv.FilterΔ

convmatrix = convmatrix(weight=conv.Filter.reshape(weightshape), outshape=convout.shape)
convmatrixout = convmatrix.forward(input)
convmatrixinerror = convmatrix.backward(1 - convmatrixout)
convmatrixdelta = conv.FilterΔ

print(convdelta == convmatrixdelta)