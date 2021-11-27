import numpy as np
from .activation import *

class convmatrix:
  def __init__(self, filtershape, learningrate, activation=identity, inshape=None):
    self.learningrate, self.activation = learningrate, activation
    self.inshape = inshape

    self.filtershape = filtershape
    self.Filter = np.random.uniform(-0.25, 0.25, (self.filtershape))
    self.biases = np.zeros(self.filtershape[0])

    '''
    if inshape is not None:
      if len(inshape) == 2:
        inshape = tuple([self.filtershape[0]]) + inshape

      self.outshape = tuple([filtershape[0], inshape[1] - filtershape[1] + 1, inshape[2] - filtershape[2] + 1])
      self.out = np.zeros(self.outshape)
      
      self.filtermatrix = np.zeros((self.filtershape[0], np.prod(self.inshape), np.prod(self.outshape)))
      for i in range(np.prod(self.outshape)):
        x = np.zeros((self.filtershape[0], self.filtershape[1] * self.inshape[1]))
        for j in range(self.filtershape[1]):
          x[j * self.inshape[1] : (j + self.filtershape[1]) * self.inshape[1]] = self.Filter[:, j, :]
        self.filtermatrix[:, i : i + (self.filtershape[1] * self.inshape[1]), i] = x
    '''

  def __copy__(self):
    return type(self)(self.filtershape, self.learningrate, self.activation, self.padding, self.stride, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape
    if len(inshape) == 2:
      inshape = tuple([self.filtershape[0]]) + inshape
    self.outshape = tuple([self.filtershape[0], inshape[1] - self.filtershape[1] + 1, inshape[2] - self.filtershape[2] + 1])
    self.out = np.zeros(self.outshape)
      
    self.filtermatrix = np.zeros((self.filtershape[0], np.prod(self.inshape), np.prod(self.outshape)))
    for i in range(np.prod(self.outshape) - inshape + self.filtershape[1]):
      x = np.zeros((self.filtershape[0], self.filtershape[1] * self.inshape[1]))
      for j in range(self.filtershape[1]):
        x[:, j * self.inshape[1] : (j * self.inshape[1]) + self.filtershape[1]] = self.Filter[:, :, j]
      self.filtermatrix[:, i : i + (self.filtershape[1] * self.inshape[1]), i] = x

    return self.outshape

  def SetSlopeForLeakyReLU(self, Slope):
    LeakyReLU.Slope = Slope

  def forward(self, input):
    self.input = input
    if(input.ndim == 2):
      self.input = np.stack(([self.input] * self.filtershape[0]))

    self.out = self.filtermatrix.T @ self.input.flatten() + self.biases
    self.out = self.activation.forward(self.out)
    return self.out.reshape(self.outshape)
  
  def backward(self, outError):
    FilterΔ = np.zeros(self.filtershape)
    
    outGradient = self.activation.backward(self.out) * outError

    for i in range(0, self.outshape[1], self.stride[0]):
      for j in range(0, self.outshape[2], self.stride[1]):
        FilterΔ += self.input[:, i : i + self.filtershape[1], j : j + self.filtershape[2]] * outGradient[:, i, j][:, np.newaxis, np.newaxis]
    
    self.biases += np.sum(outGradient, axis=(1, 2)) * self.learningrate
    self.Filter += FilterΔ * self.learningrate

    # in Error
    RotFilter = np.rot90(np.rot90(self.Filter))
    PaddedError = np.pad(outError, self.filtershape[1] - 1, mode='constant')[self.filtershape[1] - 1 : self.filtershape[0] + self.filtershape[1] - 1, :, :]
    
    self.inError = np.zeros(self.inshape)
    if np.ndim(self.inError) == 3:
      for i in range(int(self.inshape[1] / self.stride[0])):
        for j in range(int(self.inshape[2] / self.stride[1])):
         self.inError[:, i * self.stride[0], j * self.stride[1]] = np.sum(np.multiply(RotFilter, PaddedError[:, i:i + self.filtershape[1], j:j + self.filtershape[2]]), axis=(1, 2))
       
    if np.ndim(self.inError) == 2:
      for i in range(int(self.inshape[0] / self.stride[0])):
        for j in range(int(self.inshape[1] / self.stride[1])):
         self.inError[i * self.stride[0], j * self.stride[1]] = np.sum(np.multiply(RotFilter, PaddedError[:, i:i + self.filtershape[1], j:j + self.filtershape[2]]))

    return self.inError