import numpy as np

class Neat:
  def __init__(self, OutSize, MutationRate):
    self.OutSize, self.MutationRate = InSize, OutSize, MutationRate
    self.OutBias = np.zeros(OutSize)
    self.Weights = np.zeros(3)

  def ForwardProp(self, In):
    self.In, self.InSize = In, len(In)
    self.Out = np.zeros(self.OutSize)
    