import numpy as np
# POOL doesn't work with stride != 1

class POOL: 
  def __init__(self, Stride, PoolShape):
    self.Stride, self.PoolShape = Stride, PoolShape
    
  def ForwardProp(self, Input):
    self.Input, InputShape = Input, Input.shape
    OutputWidth = int(len(Input[0, 0]) / 2)
    OutputHeight = int(len(Input[0]) / 2)
    self.OutputArray = np.zeros((len(Input), OutputHeight, OutputWidth))
    for i in range(OutputHeight, InputShape[1]):
      for j in range(OutputWidth, InputShape[2]):
        self.OutputArray[:, i, j] = np.amax(Input[:, i * self.Stride : i * self.Stride + 2, j * self.Stride : j * self.Stride + 2], axis=(1, 2))
    return self.OutputArray

  def BackProp(self, CurrentMaxPoolingLayerError):
    return np.repeat(np.repeat(CurrentMaxPoolingLayerError, 2, axis=1), 2, axis=2) * np.equal(np.repeat(np.repeat(self.OutputArray, 2, axis=1), 2, axis=2), self.Input).astype(int)