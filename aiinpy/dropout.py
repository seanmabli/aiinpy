import numpy as np

class dropout:
  def __init__(self, dropoutrate):
    self.dropoutrate = dropoutrate

  def __copy__(self):
    return type(self)(self.dropoutrate)

  def modelinit(self, inshape):
    return inshape

  def forward(self, input):
    self.Dropout = np.random.binomial(1, self.dropoutrate, size=input.shape)
    self.Dropout = np.where(self.Dropout == 0, 1, 0)
    return self.Dropout * input

  def backward(self, outError):
    return self.Dropout * outError  
    
  def changeDropoutRate(self, NewRate):
    self.dropoutrate = NewRate