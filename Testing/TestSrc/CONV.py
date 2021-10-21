import numpy as np
from .Activation import ApplyActivation, ActivationDerivative

class CONV:
  def __init__(self, FilterShape, LearningRate, Activation='None', Padding=False, Stride=(1, 1)):
    self.Filter = np.random.uniform(-0.25, 0.25, (FilterShape))
    self.Bias = np.zeros(FilterShape[0])
    self.LearningRate, self.Activation, self.Padding, self.Stride, self.NumOfFilters, self.FilterShape = LearningRate, Activation, Padding, Stride, FilterShape[0], FilterShape

  def SetSlopeForLeakyReLU(self, Slope):
    LeakyReLU.Slope = Slope

  def ForwardProp(self, Input):
    self.Input = Input
    self.InputShape = self.Input.shape
    if(Input.ndim == 2):
      self.Input = np.stack(([self.Input] * self.NumOfFilters))
    if (self.Padding == True):
      self.Input = np.pad(self.Input, int((len(self.Filter[0]) - 1) / 2), mode='constant')[1 : self.NumOfFilters + 1]

    self.OutWidth = int((len(self.Input[0, 0]) - (len(self.Filter[0, 0]) - 1)) / self.Stride[0])
    self.OutHeight = int((len(self.Input[0]) - (len(self.Filter[0]) - 1)) / self.Stride[1])

    self.Output = np.zeros((self.NumOfFilters, self.OutHeight, self.OutWidth))

    for i in range(0, self.OutHeight, self.Stride[0]):
      for j in range(0, self.OutWidth, self.Stride[1]):
        self.Output[:, i, j] = np.sum(np.multiply(self.Input[:, i : i + 3, j : j + 3], self.Filter), axis=(1, 2))

    self.Output += self.Bias[:, np.newaxis, np.newaxis]
    self.Output = ApplyActivation(self.Output, self.Activation)

    return self.Output
  
  def BackProp(self, OutError):
    FilterΔ = np.zeros((self.NumOfFilters, 3, 3))
    
    OutGradient = ActivationDerivative(self.Output, self.Activation) * OutError

    for i in range(0, self.OutHeight, self.Stride[0]):
      for j in range(0, self.OutWidth, self.Stride[1]):
        FilterΔ += self.Input[:, i : (i + 3), j : (j + 3)] * OutGradient[:, i, j][:, np.newaxis, np.newaxis]
    
    self.Bias += np.sum(OutGradient, axis=(1, 2)) * self.LearningRate
    self.Filter += FilterΔ * self.LearningRate

    # Input Error
    self.RotFilter = np.rot90(np.rot90(self.Filter))
    
    y = np.pad(OutError, self.FilterShape[1] - 1, mode='constant')[self.FilterShape[1] - 1 : self.NumOfFilters + self.FilterShape[1] - 1, :, :]
    
    self.InputError = np.zeros(self.InputShape)
    if np.ndim(self.InputError) == 3:
      for i in range(int(self.InputShape[1] / self.Stride[0])):
        for j in range(int(self.InputShape[2] / self.Stride[1])):
         self.InputError[:, i * self.Stride[0], j * self.Stride[1]] = np.sum(np.multiply(self.RotFilter, y[:, i:i + 3, j:j + 3]), axis=(1, 2))
       
    if np.ndim(self.InputError) == 2:
      for i in range(int(self.InputShape[0] / self.Stride[0])):
        for j in range(int(self.InputShape[1] / self.Stride[1])):
         self.InputError[i * self.Stride[0], j * self.Stride[1]] = np.sum(np.multiply(self.RotFilter, y[:, i:i + 3, j:j + 3]))

    return self.InputError