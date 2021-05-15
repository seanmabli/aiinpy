import numpy as np

class CONV:
  def __init__(self, FilterShape, LearningRate, Padding='None', Stride=(1, 1)):
    self.Filter = np.random.uniform(-0.25, 0.25, (FilterShape))
    self.NumOfFilters = 4
    self.LearningRate, self.Padding, self.Stride = LearningRate, Padding, Stride

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
    return self.OutputArray
  
  def BackProp(self, ConvolutionError):
    FilterGradients = np.zeros((self.NumOfFilters, 3, 3))
    for i in range(self.OutputHeight):
      for j in range(self.OutputWidth):
        FilterGradients += self.InputImage[:, i : (i + 3), j : (j + 3)] * ConvolutionError[:, i, j][:, np.newaxis, np.newaxis]
    self.Filter += FilterGradients * self.LearningRate