import numpy as np
from .ActivationFunctions import ForwardProp, BackProp
import sys

class CONV:
  def __init__(self, FilterShape, LearningRate, Activation='None', Padding=False, Stride=(1, 1), DropoutRate=0):
    self.Filter = np.random.uniform(-0.25, 0.25, (FilterShape))
    self.Bias = np.zeros(FilterShape[0])
    self.NumOfFilters = FilterShape[0]
    self.LearningRate, self.Activation, self.Padding, self.Stride, self.DropoutRate, self.FilterShape = LearningRate, Activation, Padding, Stride, DropoutRate, FilterShape

  def SetSlopeForLeakyReLU(self, Slope):
    LeakyReLU.Slope = Slope
    
  def ChangeDropoutRate(self, NewRate):
    self.DropoutRate = NewRate

  def ForwardProp(self, Input):
    self.Input = Input
    self.InputShape = self.Input.shape
    if(Input.ndim == 2):
      self.Input = np.stack(([self.Input] * self.NumOfFilters))
    if (self.Padding == True):
      self.Input = np.pad(self.Input, int((len(self.Filter[0]) - 1) / 2), mode='constant')[1 : self.NumOfFilters + 1]

    self.OutputWidth = int((len(self.Input[0, 0]) - (len(self.Filter[0, 0]) - 1)) / self.Stride[0])
    self.OutputHeight = int((len(self.Input[0]) - (len(self.Filter[0]) - 1)) / self.Stride[1])

    self.Output = np.zeros((self.NumOfFilters, self.OutputHeight, self.OutputWidth))

    for i in range(0, self.OutputWidth, self.Stride[0]):
      for j in range(0, self.OutputHeight, self.Stride[1]):
        self.Output[:, i, j] = np.sum(np.multiply(self.Input[:, i : i + 3, j : j + 3], self.Filter), axis=(1, 2))

    self.Output = ForwardProp(self.Output, self.Activation)

    self.Dropout = np.random.binomial(1, self.DropoutRate, size=self.Output.shape)
    self.Dropout = np.where(self.Dropout == 0, 1, 0)
    self.Output *= self.Dropout

    return self.Output
  
  def BackProp(self, OutputError):
    FilterGradients = np.zeros((self.NumOfFilters, 3, 3))
    OutputError *= self.Dropout
    
    Derivative = BackProp(self.Output, self.Activation)

    x = OutputError * Derivative

    for i in range(0, self.OutputWidth, self.Stride[0]):
      for j in range(0, self.OutputHeight, self.Stride[1]):
        FilterGradients += self.Input[:, i : (i + 3), j : (j + 3)] * x[:, i, j][:, np.newaxis, np.newaxis]
    
    self.Filter += FilterGradients * self.LearningRate

    self.RotFilter = np.rot90(np.rot90(self.Filter))
    z = self.FilterShape[1] - 1
    y = np.pad(OutputError, z, mode='constant')[z : self.NumOfFilters + z, :, :]

    self.PreviousError = np.zeros(self.InputShape)
    if np.ndim(self.PreviousError) == 3:
      for i in range(self.InputShape[1]):
        for j in range(self.InputShape[2]):
         self.PreviousError[:, i, j] = np.sum(np.multiply(self.RotFilter, y[:, i:i + 3, j:j + 3]), axis=(1, 2))
       
    if np.ndim(self.PreviousError) == 2:
      for i in range(self.InputShape[0]):
        for j in range(self.InputShape[1]):
         self.PreviousError[i, j] = np.sum(np.multiply(self.RotFilter, y[:, i:i + 3, j:j + 3]))
         
    return self.PreviousError