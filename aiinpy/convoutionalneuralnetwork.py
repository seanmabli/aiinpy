import numpy as np
from activationfunctions import Sigmoid, DerivativeOfSigmoid, StableSoftMax, DerivativeOfStableSoftMax, ReLU, DerivativeOfReLU, Tanh, DerivativeOfTanh

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

class POOL: 
  def __init__(self, Stride):
    self.Stride = Stride
    
  def ForwardProp(self, InputArray):
    self.InputArray = InputArray
    OutputWidth = int(len(InputArray[0, 0]) / 2)
    OutputHeight = int(len(InputArray[0]) / 2)
    NumberOfFilters = len(InputArray)
    self.OutputArray = np.zeros((NumberOfFilters, OutputHeight, OutputWidth))
    for FilterNumber in range(NumberOfFilters):
      for i in range(OutputHeight):
        for j in range(OutputWidth):
          self.OutputArray[FilterNumber, i, j] = np.max(InputArray[FilterNumber, i * self.Stride : i * self.Stride + 2, j * self.Stride : j * self.Stride + 2])
    return self.OutputArray

  def BackProp(self, CurrentMaxPoolingLayerError):
    OutputWidth = len(self.InputArray[0, 0])
    OutputHeight = len(self.InputArray[0])
    NumberOfFilters = len(self.InputArray)
    PreviousConvolutionalLayerError = np.zeros(self.InputArray.shape)
    for FilterNumber in range(NumberOfFilters):
      for i in range(OutputHeight):
        for j in range(OutputWidth):
          if(self.OutputArray[FilterNumber, int(i / 2), int(j / 2)] == self.InputArray[FilterNumber, i, j]):
            PreviousConvolutionalLayerError[FilterNumber, i, j] = CurrentMaxPoolingLayerError[FilterNumber, int(i / 2), int(j / 2)] 
          else:
            PreviousConvolutionalLayerError[FilterNumber, i, j] = 0
    return PreviousConvolutionalLayerError