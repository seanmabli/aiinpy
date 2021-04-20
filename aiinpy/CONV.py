import numpy as np

class CONV:
  def __init__(self, FilterShape, LearningRate):
    self.Filter = np.random.uniform(-0.25, 0.25, (FilterShape))
    self.LearningRate = LearningRate

  def ForwardProp(self, InputImage):
    self.InputImage = InputImage
    self.OutputWidth = len(InputImage[0]) - (len(self.Filter[0, 0]) - 1)
    self.OutputHeight = len(InputImage) - (len(self.Filter[0]) - 1)
    NumberOfFilters = len(self.Filter)
    OutputArray = np.zeros((NumberOfFilters, self.OutputHeight, self.OutputWidth))
    for FilterNumber in range(NumberOfFilters):
      for i in range(self.OutputHeight):
        for j in range(self.OutputWidth):
          OutputArray[FilterNumber, i, j] = np.sum(np.multiply(InputImage[i : i + 3, j : j + 3], self.Filter[FilterNumber, :, :]))
    return OutputArray
  
  def BackProp(self, ConvolutionError):
    NumberOfFilters = len(self.Filter)
    FilterGradients = np.zeros((NumberOfFilters, 3, 3))
    for FilterNumber in range(NumberOfFilters):
      for i in range(self.OutputHeight):
        for j in range(self.OutputWidth):
          FilterGradients[FilterNumber, :, :] += self.InputImage[i : (i + 3), j : (j + 3)] * ConvolutionError[FilterNumber, i, j]
    self.Filter += FilterGradients * self.LearningRate
