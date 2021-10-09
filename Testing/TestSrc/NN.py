import numpy as np
from .Activation import *

class NN:
  def __init__(self, InShape, OutShape, Activation, LearningRate, WeightsInit=(-1, 1), BiasesInit=(0, 0), DropoutRate=0):
    self.Weights = np.random.uniform(WeightsInit[0], WeightsInit[1], (np.prod(InShape), np.prod(OutShape)))
    self.Biases = np.random.uniform(BiasesInit[0], BiasesInit[1], np.prod(OutShape))
    self.InShape, self.OutShape = InShape, OutShape
    self.Activation, self.LearningRate, self.DropoutRate = Activation, LearningRate, DropoutRate

  def ChangeDropoutRate(self, NewRate):
    self.DropoutRate = NewRate
    
  def ForwardProp(self, In):
    self.In = In.flatten()
    self.Out = np.transpose(self.Weights) @ self.In + self.Biases
  
    # Apply Activation Function
    self.Out = ApplyActivation(self.Out, self.Activation)

    self.Dropout = np.random.binomial(1, self.DropoutRate, size=len(self.Out))
    self.Dropout = np.where(self.Dropout == 0, 1, 0)
    self.Out *= self.Dropout

    return self.Out.reshape(self.OutShape)

  def BackProp(self, OutError):
    OutError = OutError.flatten()
    OutError *= self.Dropout

    # Apply Activation Function Derivative
    OutGradient = np.multiply(ActivationDerivative(self.Out, self.Activation), OutError)
      
    # Calculate Current Layer Error
    InputError = self.Weights @ OutError
      
    # Apply Deltas To The Weights And Biases
    self.Biases += OutGradient * self.LearningRate
    self.Weights += np.outer(np.transpose(self.In), OutGradient) * self.LearningRate
    return InputError.reshape(self.InShape)