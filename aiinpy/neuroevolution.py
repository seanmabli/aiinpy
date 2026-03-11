import numpy as np
from copy import copy

class neuroevolution:
  def __init__(self, inshape, outshape, PopulationSize, model):
    self.inshape, self.outshape, self.PopulationSize = inshape, outshape, PopulationSize
    
    self.model = np.zeros((PopulationSize, len(model)), dtype=object)
    for i in range(PopulationSize):
      inshape = self.inshape
      for j in range(len(model)):
        self.model[i, j] = copy(model[j])
        inshape = self.model[i, j].modelinit(inshape)

  def forwardmulti(self, input):
    out = np.zeros((self.PopulationSize, self.outshape))
    for i in range(self.PopulationSize):
      hid = input
      for j in range(self.model.shape[1]):
        hid = self.model[i, j].forward(hid)
      out[i] = hid
    return out

  def forwardsingle(self, input, Player):
    for j in range(self.model.shape[1]):
      input = self.model[Player, j].forward(input)
    return input

  def mutate(self, FavorablePlayer):
    Favorablemodel = self.model[FavorablePlayer]

    for i in range(self.model.shape[1]):
      self.model[0, i].weights = Favorablemodel[i].weights.copy()
      self.model[0, i].biases = Favorablemodel[i].biases.copy()

    for i in range(1, self.PopulationSize):
      for j in range(self.model.shape[1]):
        w = Favorablemodel[j].weights.copy()
        b = Favorablemodel[j].biases.copy()
        lr = self.model[i, j].learningrate

        wmask = np.random.random(w.shape) < lr
        w[wmask] = np.random.uniform(-1, 1, np.count_nonzero(wmask))
        self.model[i, j].weights = w

        bmask = np.random.random(b.shape) < lr
        b[bmask] = np.random.uniform(-1, 1, np.count_nonzero(bmask))
        self.model[i, j].biases = b
