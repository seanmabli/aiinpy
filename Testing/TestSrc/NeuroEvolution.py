import numpy as np
from alive_progress import alive_bar

class neuroevolution:
  def __init__(self, InSize, OutSize, PopulationSize, Model):
    self.InSize, self.OutSize, self.PopulationSize = InSize, OutSize, PopulationSize
    self.Weights = np.random.uniform(-1, 1, (PopulationSize, InSize, OutSize))
    self.Biases = np.random.uniform(0, 0, (PopulationSize, OutSize))
    self.Model = np.array([Model] * PopulationSize)

  def forwardprop(self, In):
    Out = np.zeros((self.PopulationSize, self.OutSize))
    for i in range(self.PopulationSize):
      Hid = In
      for j in range(self.Model.shape[1]):
        Hid = self.Model[i, j].forwardprop(Hid)
      Out[i] = Hid
    return Out

  def mutate(self, FavorablePlayer):
    FavorableModel = self.Model[FavorablePlayer]

    for i in range(self.PopulationSize):
      self.Model[i] = FavorableModel
      for j in range(self.Model.shape[1]):
        self.Model[i, j].mutate(FavorableModel[j])