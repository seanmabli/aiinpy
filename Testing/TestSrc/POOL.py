import numpy as np

class POOL: 
  def __init__(self, Stride, FilterShape, Type):
    self.Stride, self.FilterShape, self.Type = Stride, FilterShape, Type
    
  def ForwardProp(self, In):
    self.In = In
    self.Out = np.zeros((len(In), int(len(In[0]) / 2), int(len(In[0, 0]) / 2)))
    if self.Type == 'Max':
      for i in range(self.Out.shape[1]):
        for j in range(self.Out.shape[2]):
          self.Out[:, i, j] = np.amax(In[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    elif self.Type == 'Min':
      for i in range(self.Out.shape[1]):
        for j in range(self.Out.shape[2]):
          self.Out[:, i, j] = np.amin(In[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    elif self.Type == 'Mean':
      for i in range(self.Out.shape[1]):
        for j in range(self.Out.shape[2]):
          self.Out[:, i, j] = np.mean(In[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    elif self.Type == 'Sum':
      for i in range(self.Out.shape[1]):
        for j in range(self.Out.shape[2]):
          self.Out[:, i, j] = np.sum(In[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    return self.Out

  def BackProp(self, OutError):
    if self.Type == 'Max' or self.Type == 'Min':
      return np.repeat(np.repeat(OutError, 2, axis=1), 2, axis=2) * np.equal(np.repeat(np.repeat(self.Out, 2, axis=1), 2, axis=2), self.In).astype(int)
    elif self.Type == 'Mean' or self.Type == 'Sum':
      return np.repeat(np.repeat(OutError, 2, axis=1), 2, axis=2) * np.repeat(np.repeat(self.Out, 2, axis=1), 2, axis=2)