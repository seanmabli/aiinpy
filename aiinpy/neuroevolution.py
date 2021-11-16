import numpy as np
from .activation import *

class neuroevolution:
  def __init__(self, InSize, OutSize, MutationRate, PopulationSize, Activation):
    self.Weights = np.random.uniform(-1, 1, (PopulationSize, InSize, OutSize))
    self.Biases = np.random.uniform(0, 0, (PopulationSize, OutSize))
    self.MutationRate, self.PopulationSize, self.Activation = MutationRate, PopulationSize, Activation

  def forward(self, In):
    self.Out = np.multiply(In, np.transpose(self.Weights, axes=(2, 0, 1)))
    self.Out = np.transpose(np.sum(self.Out, axis=2), axes=(1, 0)) + self.Biases
    return ApplyActivation(self.Out, self.Activation)

  def Mutate(self, FavorablePlayer):
    # Copy Weights and Biases
    self.Weights = np.array([self.Weights[FavorablePlayer]] * self.PopulationSize)
    self.Biases = np.array([self.Biases[FavorablePlayer]] * self.PopulationSize)

    # Mutate Weights & Biases
    self.Weights *= np.random.choice([0, 1], self.Weights.shape, p=[self.MutationRate, 1 - self.MutationRate])
    self.Weights = np.where(self.Weights == 0, np.random.uniform(-1, 1, self.Weights.shape), self.Weights)

    self.Biases *= np.random.choice([0, 1], self.Biases.shape, p=[self.MutationRate, 1 - self.MutationRate])
    self.Biases = np.where(self.Biases == 0, np.random.uniform(-1, 1, self.Biases.shape), self.Biases)