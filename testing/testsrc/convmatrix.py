import numpy as np
from .binarystep import binarystep
from .gaussian import gaussian
from .identity import identity
from .leakyrelu import leakyrelu
from .mish import mish
from .relu import relu
from .selu import selu
from .sigmoid import sigmoid
from .silu import silu
from .softmax import softmax
from .softplus import softplus
from .stablesoftmax import stablesoftmax
from .tanh import tanh

class convmatrix:
  def __init__(self, filtershape, learningrate, activation=identity(), inshape=None):
    self.learningrate, self.activation = learningrate, activation
    self.inshape = inshape
    
    if len(filtershape) == 2:
      filtershape = tuple([1]) + filtershape
    self.filtershape = filtershape

    if inshape is not None:
      if len(inshape) == 2:
        inshape = tuple([filtershape[0]]) + inshape

      self.bias = np.zeros(filtershape[0])

      self.outshape = tuple([filtershape[0], inshape[1] - filtershape[1] + 1, inshape[2] - filtershape[2] + 1])
      self.out = np.zeros(self.outshape)

      self.filtermatrix = np.zeros((np.prod(inshape), np.prod(self.outshape)))
      x = np.zeros(inshape)
      for filter in range(self.outshape[0]):
        self.filter = np.random.uniform(-0.25, 0.25, filtershape[1:])
        for i in range(self.outshape[1]):
          for j in range(self.outshape[2]):
            x[:, i : i + filtershape[1], j : j + filtershape[2]] = self.filter
            self.filtermatrix[:, i * self.outshape[2] + j] = x.flatten()

      print(self.filtermatrix)

  def modelinit(self, inshape):
    if len(inshape) == 2:
      inshape = tuple([self.filtershape[0]]) + inshape

    flattenweight = np.random.uniform(-0.25, 0.25, self.filtershape[2])
    for i in range(1, self.filtershape[1]):
      flattenweight = np.append(flattenweight, np.zeros(inshape[1] - self.filtershape[2]))
      flattenweight = np.append(flattenweight, np.random.uniform(-0.25, 0.25, self.filtershape[2]))

    self.filter = []
    i = 0
    while i + len(flattenweight) <= np.prod(inshape):
      y = np.zeros((np.prod(inshape)))
      y[i : i + len(flattenweight)] = flattenweight
      self.filter.append(y)
      if (np.prod(inshape) - (i + len(flattenweight))) % inshape[0] == 0:
        i += self.filtershape[1]
      else:
        i += 1
    self.filter = np.array(self.filter)
    self.bias = np.zeros(self.filtershape[0])
    
    self.outshape = tuple([self.filtershape[0], inshape[1] - self.filtershape[1] + 1, inshape[2] - self.filtershape[2] + 1])
    return self.outshape

  def forward(self, input):
    self.input = input
    print(self.filter.shape)
    self.out = self.filter @ input.flatten() + self.bias

    self.out = self.activation.forward(self.out)

    return self.out.reshape(self.outshape)

  def backward(self, outerror):
    outerror = outerror.flatten()

    outgradient = self.activation.backward(self.out) * outerror

    inputerror = np.flip((self.filter.T @ outerror).reshape(self.inshape), axis=1)

    self.filter += np.outer(self.input.flatten().T, outgradient).reshape(self.filter.shape)
    return inputerror