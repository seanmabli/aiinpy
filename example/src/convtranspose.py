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

class convtranspose:
  def __init__(self, inshape, filtershape, learningrate, activation, padding=False, stride=(1, 1)):
    self.inshape, self.filtershape, self.learningrate, self.activation, self.padding, self.stride = inshape, filtershape, learningrate, activation, padding, stride
    if len(inshape) == 2:
      inshape = tuple([self.filtershape[0]]) + inshape
    self.outshape = np.array([filtershape[0], ((inshape[1] + 1) * filtershape[1]) / stride[0], ((inshape[2] + 1) * filtershape[2]) / stride[1]], dtype=np.int)
    self.out = np.zeros(self.outshape)

    self.Filter = np.random.uniform(-0.25, 0.25, (self.filtershape))
    self.bias = np.zeros(self.filtershape[0])

  def modelinit(self, inshape):
    return self.outshape

  def forward(self, input):
    self.input = input
    if(input.ndim == 2):
      self.input = np.stack(([self.input] * self.filtershape[0]))

    self.out = np.zeros(self.outshape)
    for i in range(0, self.inshape[1]):
      for j in range(0, self.inshape[2]):
        self.out[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[1], j * self.stride[1] : j * self.stride[1] + self.filtershape[2]] += self.input[:, i, j][:, np.newaxis, np.newaxis] * self.Filter

    self.out += self.bias[:, np.newaxis, np.newaxis]
    self.out = self.activation.forward(self.out)

    if self.padding == True:
      self.out = self.out[:, 1 : self.outshape[1] - 1, 1 : self.outshape[2] - 1]

    return self.out

  def backward(self, outError):
    FilterΔ = np.zeros(self.filtershape)

    outGradient = self.activation.backward(self.out) * outError
    outGradient = np.pad(outGradient, 1, mode='constant')[1 : self.filtershape[0] + 1, :, :]

    for i in range(0, self.inshape[1]):
      for j in range(0, self.inshape[2]):
        FilterΔ += self.input[:, i, j][:, np.newaxis, np.newaxis] * outGradient[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[1], j * self.stride[1] : j * self.stride[1] + self.filtershape[2]]

    self.bias += np.sum(outGradient, axis=(1, 2)) * self.learningrate
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