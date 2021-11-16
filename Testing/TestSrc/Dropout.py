import numpy as np

class dropout:
  def __init__(self, DropoutRate):
    self.DropoutRate = DropoutRate

  def __copy__(self):
    return type(self)(self.DropoutRate)

  def forward(self, In):
    self.Dropout = np.random.binomial(1, self.DropoutRate, size=In.shape)
    self.Dropout = np.where(self.Dropout == 0, 1, 0)
    return self.Dropout * In

  def backward(self, OutError):
    return self.Dropout * OutError  
    
  def ChangeDropoutRate(self, NewRate):
    self.DropoutRate = NewRate