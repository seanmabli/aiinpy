import numpy as np
from .activation import *

class nn:
  def __init__(self, InShape, OutShape, Activation, LearningRate, WeightsInit=(-1, 1), BiasesInit=(0, 0), DropoutRate=0):
    self.Weights = np.random.uniform(WeightsInit[0], WeightsInit[1], (np.prod(InShape), np.prod(OutShape)))
    self.Biases = np.random.uniform(BiasesInit[0], BiasesInit[1], np.prod(OutShape))
    self.InShape, self.OutShape, self.Activation, self.LearningRate = InShape, OutShape, Activation, LearningRate
    
  def forwardprop(self, In):
    self.In = In.flatten()
    self.Out = self.Weights.T @ self.In + self.Biases
  
    self.Out = ApplyActivation(self.Out, self.Activation)

    return self.Out.reshape(self.OutShape)

  def backprop(self, OutError):
    OutError = OutError.flatten()
    
    OutGradient = ActivationDerivative(self.Out, self.Activation) * OutError
      
    InputError = self.Weights @ OutError
      
    self.Biases += OutGradient * self.LearningRate
    self.Weights += np.outer(self.In.T, OutGradient) * self.LearningRate
    return InputError.reshape(self.InShape)

  def mutate(self, FavorableModel):
    self.Weights = FavorableModel.Weights
    self.Biases = FavorableModel.Biases

    self.Weights *= np.random.choice([0, 1], self.Weights.shape, p=[self.LearningRate, 1 - self.LearningRate])
    self.Weights = np.where(self.Weights == 0, np.random.uniform(-1, 1, self.Weights.shape), self.Weights)

    self.Biases *= np.random.choice([0, 1], self.Biases.shape, p=[self.LearningRate, 1 - self.LearningRate])
    self.Biases = np.where(self.Biases == 0, np.random.uniform(-1, 1, self.Biases.shape), self.Biases)