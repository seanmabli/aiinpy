import numpy as np

class Dropout:
  def __init__(self, DropoutRate):
    self.DropoutRate = DropoutRate
  
  def ForwardProp(self, In):
    self.Dropout = np.random.binomial(1, self.DropoutRate, size=In.shape)
    self.Dropout = np.where(self.Dropout == 0, 1, 0)
    return self.Dropout * In

  def BackProp(self, OutError):
    return self.Dropout * OutError