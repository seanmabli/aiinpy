import numpy as np
from ActivationFunctions import Sigmoid, Tanh, ReLU, LeakyReLU, StableSoftMax
Sigmoid, Tanh, ReLU, LeakyReLU, StableSoftMax = Sigmoid(), Tanh(), ReLU(), LeakyReLU(), StableSoftMax()
import sys

class CONV:
  def __init__(self, FilterShape, LearningRate, Activation='None', Padding='None', Stride=(1, 1), DropoutRate=0):
    self.Filter = np.random.uniform(-0.25, 0.25, (FilterShape))
    self.Bias = np.zeros(FilterShape[0])
    self.NumOfFilters = FilterShape[0]
    self.LearningRate, self.Activation, self.Padding, self.Stride, self.DropoutRate = LearningRate, Activation, Padding, Stride, DropoutRate

  def SetSlopeForLeakyReLU(self, Slope):
    LeakyReLU.Slope = Slope
    
  def ChangeDropoutRate(self, NewRate):
    self.DropoutRate = NewRate

  def ForwardProp(self, Input):
    if (self.Padding == 'None'):
      if(Input.ndim == 2):
        self.Input = np.stack(([Input] * self.NumOfFilters))
      else:
        self.Input = Input
    if (self.Padding == 'Same'):
      if(Input.ndim == 2):
        self.Input = np.stack(([np.pad(Input, int((len(self.Filter[0]) - 1) / 2), mode='constant')] * self.NumOfFilters))
      else:
        self.Input = np.pad(Input, int((len(self.Filter[0]) - 1) / 2), mode='constant')[1 : self.NumOfFilters + 1]
    
    self.OutputWidth = int((len(self.Input[0, 0]) - (len(self.Filter[0, 0]) - 1)) / self.Stride[0])
    self.OutputHeight = int((len(self.Input[0]) - (len(self.Filter[0]) - 1)) / self.Stride[1])

    self.Output = np.zeros((self.NumOfFilters, self.OutputHeight, self.OutputWidth))

    for i in range(0, self.OutputWidth, self.Stride[0]):
      for j in range(0, self.OutputHeight, self.Stride[1]):
        self.Output[:, i, j] = np.sum(np.multiply(self.Input[:, i : i + 3, j : j + 3], self.Filter), axis=(1, 2))

    # self.Output += self.Bias[:, np.newaxis, np.newaxis]

    if self.Activation == 'Sigmoid': self.Output = Sigmoid.Sigmoid(self.Output)
    if self.Activation == 'Tanh': self.Output = Tanh.Tanh(self.Output)
    if self.Activation == 'ReLU': self.Output = ReLU.ReLU(self.Output)
    if self.Activation == 'LeakyReLU': self.Output = LeakyReLU.LeakyReLU(self.Output)
    if self.Activation == 'StableSoftMax': self.Output = StableSoftMax.StableSoftMax(self.Output)
    if self.Activation == 'None': self.Output = self.Output

    self.Dropout = np.random.binomial(1, self.DropoutRate, size=self.Output.shape)
    self.Dropout = np.where(self.Dropout == 0, 1, 0)
    self.Output *= self.Dropout

    return self.Output
  
  def BackProp(self, ConvolutionError):
    FilterGradients = np.zeros((self.NumOfFilters, 3, 3))
    ConvolutionError *= self.Dropout
    
    if self.Activation == 'Sigmoid': Derivative = Sigmoid.Derivative(self.Output)
    if self.Activation == 'Tanh': Derivative = Tanh.Derivative(self.Output)
    if self.Activation == 'ReLU': Derivative = ReLU.Derivative(self.Output)
    if self.Activation == 'LeakyReLU': Derivative = LeakyReLU.Derivative(self.Output)
    if self.Activation == 'StableSoftMax': Derivative = StableSoftMax.Derivative(self.Output)
    if self.Activation == 'None': Derivative = self.Output

    x = ConvolutionError * Derivative

    for i in range(0, self.OutputWidth, self.Stride[0]):
      for j in range(0, self.OutputHeight, self.Stride[1]):
        FilterGradients += self.Input[:, i : (i + 3), j : (j + 3)] * x[:, i, j][:, np.newaxis, np.newaxis]
    
    self.Filter += FilterGradients * self.LearningRate
    # self.Bias += np.clip(np.sum(ConvolutionError, axis=(1, 2)), -1, 1) * self.LearningRate

    for i in range(self.OutputWidth):
      for j in range():
        self.OutputError = 