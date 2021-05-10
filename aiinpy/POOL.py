import numpy as np

class POOL: 
  def __init__(self, Stride):
    self.Stride = Stride
    
  def ForwardProp(self, InputArray):
    self.InputArray = InputArray
    OutputWidth = int(len(InputArray[0, 0]) / 2)
    OutputHeight = int(len(InputArray[0]) / 2)
    NumOfFilters = len(InputArray)
    self.OutputArray = np.zeros((NumOfFilters, OutputHeight, OutputWidth))
    for FilterNumber in range(NumOfFilters):
      for i in range(OutputHeight):
        for j in range(OutputWidth):
          self.OutputArray[FilterNumber, i, j] = np.max(InputArray[FilterNumber, i * self.Stride : i * self.Stride + 2, j * self.Stride : j * self.Stride + 2])
    return self.OutputArray

  def BackProp(self, CurrentMaxPoolingLayerError):
    OutputWidth = len(self.InputArray[0, 0])
    OutputHeight = len(self.InputArray[0])
    NumOfFilters = len(self.InputArray)
    PreviousConvolutionalLayerError = np.zeros(self.InputArray.shape)
    for FilterNumber in range(NumOfFilters):
      for i in range(OutputHeight):
        for j in range(OutputWidth):
          if(self.OutputArray[FilterNumber, int(i / 2), int(j / 2)] == self.InputArray[FilterNumber, i, j]):
            PreviousConvolutionalLayerError[FilterNumber, i, j] = CurrentMaxPoolingLayerError[FilterNumber, int(i / 2), int(j / 2)] 
          else:
            PreviousConvolutionalLayerError[FilterNumber, i, j] = 0
    return PreviousConvolutionalLayerError