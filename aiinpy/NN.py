import numpy as np
from .activation import *

class nn:
  def __init__(self, InShape, OutShape, Activation, LearningRate, WeightsInit=(-1, 1), BiasesInit=(0, 0), DropoutRate=0):
    self.Weights = np.random.uniform(WeightsInit[0], WeightsInit[1], (np.prod(InShape), np.prod(OutShape)))
    self.Biases = np.random.uniform(BiasesInit[0], BiasesInit[1], np.prod(OutShape))
    self.InShape, self.OutShape, self.Activation, self.LearningRate = InShape, OutShape, Activation, LearningRate

  def ChangeDropoutRate(self, NewRate):
    self.DropoutRate = NewRate
    
  def forwardprop(self, In):
    self.In = In.flatten()
    self.Out = self.Weights.T @ self.In + self.Biases
  
    # Apply Activation Function
    self.Out = ApplyActivation(self.Out, self.Activation)

    return self.Out.reshape(self.OutShape)

  def backprop(self, OutError):
    OutError = OutError.flatten()

    # Apply Activation Function Derivative
    OutGradient = np.multiply(ActivationDerivative(self.Out, self.Activation), OutError)
      
    # Calculate Current Layer Error
    InputError = self.Weights @ OutError
      
    # Apply Deltas To The Weights And Biases
    self.Biases += OutGradient * self.LearningRate
    self.Weights += np.outer(self.In.T, OutGradient) * self.LearningRate
    return InputError.reshape(self.InShape)