import numpy as np
# POOL doesn't work with stride != 1

class POOL: 
  def __init__(self, Stride):
    self.Stride = Stride
    
  def ForwardProp(self, InputArray):
    self.InputArray = InputArray
    OutputWidth = int(len(InputArray[0, 0]) / 2)
    OutputHeight = int(len(InputArray[0]) / 2)
    self.OutputArray = np.zeros((len(InputArray), OutputHeight, OutputWidth))
    for i in range(OutputHeight):
      for j in range(OutputWidth):
        self.OutputArray[:, i, j] = np.amax(InputArray[:, i * self.Stride : i * self.Stride + 2, j * self.Stride : j * self.Stride + 2], axis=(1, 2))
    return self.OutputArray

  def BackProp(self, CurrentMaxPoolingLayerError):
    return np.repeat(np.repeat(CurrentMaxPoolingLayerError, 2, axis=1), 2, axis=2) * np.equal(np.repeat(np.repeat(self.OutputArray, 2, axis=1), 2, axis=2), self.InputArray).astype(int)