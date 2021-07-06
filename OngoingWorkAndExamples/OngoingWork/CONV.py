import numpy as np
from ActivationFunctions import ForwardProp, BackProp
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
    if (self.Padding == False):
      if(Input.ndim == 2):
        self.Input = np.stack(([Input] * self.NumOfFilters))
      else:
        self.Input = Input

    if (self.Padding == True):
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

    self.FilterFliped = np.rot90(np.rot90(self.Filter))
    y = np.pad(OutputError, 2, mode='constant')[2 : self.NumOfFilters + 2, :, :]

    self.PreviousError = np.zeros(self.Input.shape)
    for i in range(self.OutputWidth + len(self.Filter[0, 0]) - 1):
      for j in range(self.OutputHeight + len(self.Filter[0]) - 1):
       self.PreviousError[:, i, j] = np.sum(np.multiply(self.FilterFliped, y[:, i:i + 3, j:j + 3]), axis=(1, 2))

    return self.PreviousError[:, int(self.Padding) : self.OutputWidth + len(self.Filter[0, 0]) - 1 - int(self.Padding), int(self.Padding) : self.OutputHeight + len(self.Filter[0]) - 1 - int(self.Padding)]