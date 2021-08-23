import numpy as np
from .Activation import ApplyActivation, ActivationDerivative

class NN:
  def __init__(self, InSize, OutSize, Activation, LearningRate, WeightsInit=(-1, 1), BiasesInit=(0, 0)):
    self.Weights = np.random.uniform(WeightsInit[0], WeightsInit[1], (InSize, OutSize))
    self.Biases = np.random.uniform(BiasesInit[0], BiasesInit[1], (OutSize))
    self.Activation, self.LearningRate = Activation, LearningRate

  def ForwardProp(self, In):
    self.In = In
    self.Out = np.transpose(self.Weights) @ In + self.Biases
  
    # Apply Activation Function
    self.Out = ApplyActivation(self.Out, self.Activation)
    return self.Out

  def BackProp(self, OutError):
    # Apply Activation Function Derivative
    OutGradient = np.multiply(ActivationDerivative(self.Out, self.Activation), OutError)
      
    # Calculate Current Layer Error
    InputError = self.Weights @ OutError
      
    # Apply Deltas To The Weights And Biases
    self.Biases += OutGradient * self.LearningRate
    self.Weights += np.outer(np.transpose(self.In), OutGradient) * self.LearningRate
    return InputError