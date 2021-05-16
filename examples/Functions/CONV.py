import numpy as np
from ActivationFunctions import Sigmoid, DerivativeOfSigmoid
from ActivationFunctions import Tanh, DerivativeOfTanh
from ActivationFunctions import ReLU, DerivativeOfReLU
from ActivationFunctions import LeakyReLU, DerivativeOfLeakyReLU
from ActivationFunctions import StableSoftMax, DerivativeOfStableSoftMax

class CONV:
  def __init__(self, FilterShape, LearningRate, Activation='None', Padding='None', Stride=(1, 1)):
    self.Filter = np.random.uniform(-0.25, 0.25, (FilterShape))
    self.NumOfFilters = 64
    self.LearningRate, self.Activation, self.Padding, self.Stride = LearningRate, Activation, Padding, Stride

  # def SetSlopeForLeakyReLU(Slope):

  def ForwardProp(self, InputImage):
    if (self.Padding == 'None'):
      self.InputImage = np.stack(([InputImage] * self.NumOfFilters))
      self.OutputWidth = int((len(InputImage[0]) - (len(self.Filter[0, 0]) - 1)) / self.Stride[0])
      self.OutputHeight = int((len(InputImage) - (len(self.Filter[0]) - 1)) / self.Stride[1])
    if (self.Padding == 'Same'):
      self.InputImage = np.stack(([np.pad(InputImage, int((len(self.Filter[0]) - 1) / 2), mode='constant')] * self.NumOfFilters))
      self.OutputWidth = int(len(InputImage[0]) / self.Stride[0])
      self.OutputHeight = int(len(InputImage) / self.Stride[1])

    self.OutputArray = np.zeros((self.NumOfFilters, self.OutputHeight, self.OutputWidth))

    for i in range(0, self.OutputWidth, self.Stride[0]):
      for j in range(0, self.OutputHeight, self.Stride[1]):
        self.OutputArray[:, i, j] = np.sum(np.multiply(self.InputImage[:, i : i + 3, j : j + 3], self.Filter), axis=(1, 2))

    
    if self.Activation == 'Sigmoid': self.OutputArray = Sigmoid(self.OutputArray)
    if self.Activation == 'Tanh': self.OutputArray = Tanh(self.OutputArray)
    if self.Activation == 'ReLU': self.OutputArray = ReLU(self.OutputArray)
    if self.Activation == 'StableSoftMax': self.OutputArray = StableSoftMax(self.OutputArray)
    if self.Activation == 'None': self.OutputArray = self.OutputArray
    return self.OutputArray
  
  def BackProp(self, ConvolutionError):
    FilterGradients = np.zeros((self.NumOfFilters, 3, 3))
    
    if self.Activation == 'Sigmoid': Derivative = DerivativeOfSigmoid(self.OutputArray)
    if self.Activation == 'Tanh': Derivative = DerivativeOfTanh(self.OutputArray)
    if self.Activation == 'ReLU': Derivative = DerivativeOfReLU(self.OutputArray)
    if self.Activation == 'StableSoftMax': Derivative = DerivativeOfStableSoftMax(self.OutputArray)
    if self.Activation == 'None': Derivative = self.OutputArray

    for i in range(self.OutputHeight):
      for j in range(self.OutputWidth):
        FilterGradients += self.InputImage[:, i : (i + 3), j : (j + 3)] * ConvolutionError[:, i, j][:, np.newaxis, np.newaxis] * Derivative[:, i, j][:, np.newaxis, np.newaxis]
    self.Filter += FilterGradients * self.LearningRate