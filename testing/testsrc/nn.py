import numpy as np
from .activation import *

class nn:
  def __init__(self, inshape, outshape, activation, learningrate, weightsinit=(-1, 1), biasesinit=(0, 0)):
    self.weightsinit, self.biasesinit = weightsinit, biasesinit
    self.inshape, self.outshape = inshape, outshape
    self.activation, self.learningrate = activation, learningrate
    
    self.weights = np.random.uniform(weightsinit[0], weightsinit[1], (np.prod(inshape), np.prod(outshape)))
    self.biases = np.random.uniform(biasesinit[0], biasesinit[1], np.prod(outshape))
    
  def __copy__(self):
    return type(self)(self.inshape, self.outshape, self.activation, self.learningrate, self.weightsinit, self.biasesinit)

  def forward(self, input):
    self.input = input.flatten()
    self.out = self.weights.T @ self.input + self.biases
    
    self.out = self.activation.forward(self.out)

    return self.out.reshape(self.outshape)

  def backward(self, outError):
    outError = outError.flatten()
    
    outGradient = self.activation.backward(self.out) * outError
      
    inputError = self.weights @ outError
      
    self.biases += outGradient * self.learningrate
    self.weights += np.outer(self.input.T, outGradient) * self.learningrate
    return inputError.reshape(self.inshape)