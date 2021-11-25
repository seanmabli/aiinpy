import numpy as np

class dropout:
  def __init__(self, DropoutRate):
    self.DropoutRate = DropoutRate

  def __copy__(self):
    return type(self)(self.DropoutRate)

  def forward(self, input):
    self.Dropout = np.random.binomial(1, self.DropoutRate, size=input.shape)
    self.Dropout = np.where(self.Dropout == 0, 1, 0)
    return self.Dropout * input

  def backward(self, outError):
    return self.Dropout * outError  
    
  def changeDropoutRate(self, NewRate):
    self.DropoutRate = NewRate