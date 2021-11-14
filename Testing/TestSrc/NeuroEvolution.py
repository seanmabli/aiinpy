import numpy as np
from alive_progress import alive_bar
from copy import copy

class neuroevolution:
  def __init__(self, InSize, OutSize, PopulationSize, Model):
    self.InSize, self.OutSize, self.PopulationSize = InSize, OutSize, PopulationSize
    self.Weights = np.random.uniform(-1, 1, (PopulationSize, InSize, OutSize))
    self.Biases = np.random.uniform(0, 0, (PopulationSize, OutSize))
    
    self.Model = np.zeros((PopulationSize, len(Model)), dtype=object)
    for i in range(PopulationSize):
      for j in range(len(Model)):
        self.Model[i, j] = copy(Model[j])

  def forward(self, In):
    Out = np.zeros((self.PopulationSize, self.OutSize))
    for i in range(self.PopulationSize):
      Hid = In
      for j in range(self.Model.shape[1]):
        Hid = self.Model[i, j].forward(Hid)
      Out[i] = Hid
    return Out

  def mutate(self, FavorablePlayer):
    FavorableModel = self.Model[FavorablePlayer]

    for i in range(self.Model.shape[1]):
      self.Model[0, i].Weights = FavorableModel[i].Weights
      self.Model[0, i].Biases = FavorableModel[i].Biases

    for i in range(1, self.PopulationSize):
      for j in range(self.Model.shape[1]):
        self.Model[i, j].Weights = FavorableModel[j].Weights
        self.Model[i, j].Biases = FavorableModel[j].Biases

        self.Model[i, j].Weights *= np.random.choice([0, 1], self.Model[i, j].Weights.shape, p=[self.Model[i, j].LearningRate, 1 - self.Model[i, j].LearningRate])
        self.Model[i, j].Weights = np.where(self.Model[i, j].Weights == 0, np.random.uniform(-1, 1, self.Model[i, j].Weights.shape), self.Model[i, j].Weights)

        self.Model[i, j].Biases *= np.random.choice([0, 1], self.Model[i, j].Biases.shape, p=[self.Model[i, j].LearningRate, 1 - self.Model[i, j].LearningRate])
        self.Model[i, j].Biases = np.where(self.Model[i, j].Biases == 0, np.random.uniform(-1, 1, self.Model[i, j].Biases.shape), self.Model[i, j].Biases)