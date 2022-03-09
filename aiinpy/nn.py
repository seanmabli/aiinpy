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

class nn:
  def __init__(self, outshape, activation, learningrate, weightsinit=(-1, 1), biasesinit=(0, 0), inshape=None):
    self.weightsinit, self.biasesinit = weightsinit, biasesinit
    self.activation, self.learningrate = activation, learningrate
    self.inshape = inshape
    if inshape is not None:
      self.weights = np.random.uniform(weightsinit[0], weightsinit[1], (np.prod(inshape), np.prod(outshape)))
      self.biases = np.random.uniform(biasesinit[0], biasesinit[1], np.prod(outshape))
    self.outshape = outshape
    
  def __copy__(self):
    return type(self)(self.outshape, self.activation, self.learningrate, self.weightsinit, self.biasesinit, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape

    self.weights = np.random.uniform(self.weightsinit[0], self.weightsinit[1], (np.prod(inshape), np.prod(self.outshape)))
    self.biases = np.random.uniform(self.biasesinit[0], self.biasesinit[1], np.prod(self.outshape))
    return self.outshape

  def forward(self, input):
    self.input = input.flatten()
    self.out = self.activation.forward(self.weights.T @ self.input + self.biases)
    return self.out.reshape(self.outshape)

  def backward(self, outerror):
    outerror = outerror.flatten()
    outgradient = self.activation.backward(self.out) * outerror
    inputerror = self.weights @ outerror
    self.biases += outgradient * self.learningrate
    self.weights += np.outer(self.input.T, outgradient) * self.learningrate
    return inputerror.reshape(self.inshape)