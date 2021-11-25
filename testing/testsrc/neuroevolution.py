import numpy as np
from alive_progress import alive_bar
from copy import copy

class neuroevolution:
  def __init__(self, inshape, outshape, PopulationSize, Model):
    self.inshape, self.outshape, self.PopulationSize = inshape, outshape, PopulationSize
    
    self.Model = np.zeros((PopulationSize, len(Model)), dtype=object)
    for i in range(PopulationSize):
      for j in range(len(Model)):
        self.Model[i, j] = copy(Model[j])

  def forwardmulti(self, input):
    out = np.zeros((self.PopulationSize, self.outshape))
    for i in range(self.PopulationSize):
      Hid = input
      for j in range(self.Model.shape[1]):
        Hid = self.Model[i, j].forward(Hid)
      out[i] = Hid
    return out

  def forwardsingle(self, input, Player):
    out = np.zeros(self.outshape)
    for j in range(self.Model.shape[1]):
      input = self.Model[Player, j].forward(input)
    return input

  def mutate(self, FavorablePlayer):
    FavorableModel = self.Model[FavorablePlayer]

    for i in range(self.Model.shape[1]):
      self.Model[0, i].weights = FavorableModel[i].weights
      self.Model[0, i].biases = FavorableModel[i].biases

    for i in range(1, self.PopulationSize):
      for j in range(self.Model.shape[1]):
        self.Model[i, j].weights = FavorableModel[j].weights
        self.Model[i, j].biases = FavorableModel[j].biases

        self.Model[i, j].weights *= np.random.choice([0, 1], self.Model[i, j].weights.shape, p=[self.Model[i, j].learningrate, 1 - self.Model[i, j].learningrate])
        self.Model[i, j].weights = np.where(self.Model[i, j].weights == 0, np.random.uniform(-1, 1, self.Model[i, j].weights.shape), self.Model[i, j].weights)

        self.Model[i, j].biases *= np.random.choice([0, 1], self.Model[i, j].biases.shape, p=[self.Model[i, j].learningrate, 1 - self.Model[i, j].learningrate])
        self.Model[i, j].biases = np.where(self.Model[i, j].biases == 0, np.random.uniform(-1, 1, self.Model[i, j].biases.shape), self.Model[i, j].biases)