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
      self.outshape = tuple([filtershape[0], inshape[0] - filtershape[1] + 1, inshape[1] - filtershape[2] + 1])

      self.filtermatrix = np.zeros((np.prod(inshape), np.prod(self.outshape)))
      for f in range(self.outshape[0]):
        self.filter = np.random.uniform(-0.25, 0.25, filtershape[1:])
        for i in range(self.outshape[1]):
          for j in range(self.outshape[2]):
            x = np.zeros(inshape)
            x[i : i + filtershape[1], j : j + filtershape[2]] = self.filter
            self.filtermatrix[:, f * self.outshape[1] * self.outshape[2] + i * self.outshape[2] + j] = x.flatten()

  def modelinit(self, inshape):
    return self.outshape

  def forward(self, input):
    self.input = input.flatten()
    self.out = self.activation.forward(self.input @ self.filtermatrix)
    return self.out.reshape(self.outshape)

  def backward(self, outerror):
    outerror = outerror.flatten()
    outgradient = self.activation.backward(self.out) * outerror
    # inputerror = np.flip((self.filtermatrix @ outerror).reshape(self.inshape), axis=1)
    self.filtermatrix += np.outer(self.input.T, outgradient) * self.learningrate
    # return inputerror