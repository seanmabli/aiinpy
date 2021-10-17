import numpy as np

class POOL: 
  def __init__(self, Stride, Type):
    self.Stride = Stride
    
  def ForwardProp(self, In):
    self.In, InShape = In, In.shape
    OutputWidth = int(len(In[0, 0]) / 2)
    OutputHeight = int(len(In[0]) / 2)
    self.Out = np.zeros((len(In), OutputHeight, OutputWidth))
    for i in range(OutputHeight):
      for j in range(OutputWidth):
        self.Out[:, i, j] = np.amax(In[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    return self.Out

  def BackProp(self, OutError):
    return np.repeat(np.repeat(OutError, 2, axis=1), 2, axis=2) * np.equal(np.repeat(np.repeat(self.Out, 2, axis=1), 2, axis=2), self.In).astype(int)