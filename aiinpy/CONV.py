import numpy as np

class CONV:
  def __init__(self, FilterShape, LearningRate, Padding='None', Stride=(1, 1)):
    self.Filter = np.random.uniform(-0.25, 0.25, (FilterShape))
    self.NumOfFilters = 4
    self.LearningRate = LearningRate

  def ForwardProp(self, InputImage):
    self.InputImage = np.stack(([InputImage] * self.NumOfFilters))

    self.OutputWidth = len(InputImage[0]) - (len(self.Filter[0, 0]) - 1)
    self.OutputHeight = len(InputImage) - (len(self.Filter[0]) - 1)
    self.OutputArray = np.zeros((self.NumOfFilters, self.OutputHeight, self.OutputWidth))
    
    for i in range(self.OutputHeight):
      for j in range(self.OutputWidth):
        self.OutputArray[:, i, j] = np.sum(np.multiply(self.InputImage[:, i : i + 3, j : j + 3], self.Filter), axis=(1, 2))
    return self.OutputArray
  
  def BackProp(self, ConvolutionError):
    FilterGradients = np.zeros((self.NumOfFilters, 3, 3))
    for i in range(self.OutputHeight):
      for j in range(self.OutputWidth):
        FilterGradients += self.InputImage[:, i : (i + 3), j : (j + 3)] * ConvolutionError[:, i, j][:, np.newaxis, np.newaxis]
    self.Filter += FilterGradients * self.LearningRate