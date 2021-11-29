import numpy as np
from .activation import *

class convmatrix:
  def __init__(self, filtershape, filter, learningrate, activation=identity, inshape=None):
    self.learningrate, self.activation = learningrate, activation
    self.inshape = inshape

    self.filtershape = filtershape
    self.Filter = filter # np.random.uniform(-0.25, 0.25, (self.filtershape))
    self.biases = np.zeros(self.filtershape[0])

    if inshape is not None:
      self.inshape = inshape
      if len(inshape) == 2:
        inshape = tuple([self.filtershape[0]]) + inshape

      self.outshape = tuple([self.filtershape[0], inshape[1] - self.filtershape[1] + 1, inshape[2] - self.filtershape[2] + 1])
      self.outone = np.zeros(self.outshape)

      self.filtermatrix = np.zeros((np.prod(self.inshape), self.filtershape[0], np.prod(self.outshape[1:])))
      self.x = np.zeros((self.filtershape[0], self.filtershape[1] * self.inshape[1]))
      for i in range(np.prod(self.outshape[1:]) - self.inshape[1] + self.filtershape[1]):
        for j in range(self.filtershape[1]):
          self.x[:, j * self.inshape[1] : (j * self.inshape[1]) + self.filtershape[1]] = self.Filter[:, :, j]
        self.filtermatrix[i : i + (self.filtershape[1] * self.inshape[1]), :, i] = self.x.T

  def __copy__(self):
    return type(self)(self.filtershape, self.learningrate, self.activation, self.padding, self.stride, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape
    if len(inshape) == 2:
      inshape = tuple([self.filtershape[0]]) + inshape
      
    self.outshape = tuple([self.filtershape[0], inshape[1] - self.filtershape[1] + 1, inshape[2] - self.filtershape[2] + 1])
    self.outone = np.zeros(self.outshape)
      
    self.filtermatrix = np.zeros((np.prod(self.inshape), self.filtershape[0], np.prod(self.outshape[1:])))
    for i in range(np.prod(self.outshape[1:])):
      pass


    return self.outshape

  def SetSlopeForLeakyReLU(self, Slope):
    LeakyReLU.Slope = Slope

  def forward(self, input):
    if(input.ndim == 2):
      self.input = np.stack(([input] * self.filtershape[0]))

    for i in range(0, self.outshape[1]):
      for j in range(0, self.outshape[2]):
        self.outone[:, i, j] = np.sum(np.multiply(self.input[:, i : i + self.filtershape[1], j : j + self.filtershape[2]], self.Filter), axis=(1, 2))

    self.outone += self.biases[:, np.newaxis, np.newaxis]
    self.outone = self.activation.forward(self.outone)

    self.out = self.filtermatrix.T @ input.reshape(np.prod(self.inshape)) + self.biases[np.newaxis, :]
    self.out = self.activation.forward(self.out)
    self.out = self.out.reshape(self.outshape)

    print(np.sum(abs(self.out - self.outone)))
    return self.out, self.outone
  
  def backward(self, outError):
    FilterΔ = np.zeros(self.filtershape)
    
    outGradient = self.activation.backward(self.out) * outError

    for i in range(0, self.outshape[1]):
      for j in range(0, self.outshape[2]):
        FilterΔ += self.input[:, i : i + self.filtershape[1], j : j + self.filtershape[2]] * outGradient[:, i, j][:, np.newaxis, np.newaxis]
    
    self.biases += np.sum(outGradient, axis=(1, 2)) * self.learningrate
    self.Filter += FilterΔ * self.learningrate

    for i in range(np.prod(self.outshape[1:]) - self.inshape[1] + self.filtershape[1]):
      for j in range(self.filtershape[1]):
        self.x[:, j * self.inshape[1] : (j * self.inshape[1]) + self.filtershape[1]] = self.Filter[:, :, j]
      self.filtermatrix[i : i + (self.filtershape[1] * self.inshape[1]), :, i] = self.x.T

    '''
    # in Error
    RotFilter = np.rot90(np.rot90(self.Filter))
    PaddedError = np.pad(outError, self.filtershape[1] - 1, mode='constant')[self.filtershape[1] - 1 : self.filtershape[0] + self.filtershape[1] - 1, :, :]
    
    self.inError = np.zeros(self.inshape)
    if np.ndim(self.inError) == 3:
      for i in range(self.inshape[1]):
        for j in range(self.inshape[2]):
         self.inError[:, i, j] = np.sum(np.multiply(RotFilter, PaddedError[:, i:i + self.filtershape[1], j:j + self.filtershape[2]]), axis=(1, 2))
       
    if np.ndim(self.inError) == 2:
      for i in range(int(self.inshape[0])):
        for j in range(int(self.inshape[1])):
         self.inError[i, j] = np.sum(np.multiply(RotFilter, PaddedError[:, i:i + self.filtershape[1], j:j + self.filtershape[2]]))

    return self.inError
    '''