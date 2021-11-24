import numpy as np
from .activation import *

class nn:
  def __init__(self, InShape, OutShape, Activation, LearningRate, WeightsInit=(-1, 1), BiasesInit=(0, 0)):
    self.WeightsInit, self.BiasesInit = WeightsInit, BiasesInit
    self.InShape, self.OutShape = InShape, OutShape
    self.Activation, self.LearningRate = Activation, LearningRate
    
    self.Weights = np.random.uniform(WeightsInit[0], WeightsInit[1], (np.prod(InShape), np.prod(OutShape)))
    self.Biases = np.random.uniform(BiasesInit[0], BiasesInit[1], np.prod(OutShape))
    
  def __copy__(self):
    return type(self)(self.InShape, self.OutShape, self.Activation, self.LearningRate, self.WeightsInit, self.BiasesInit)

  def forward(self, In):
    self.In = In.flatten()
    self.Out = self.Weights.T @ self.In + self.Biases
    
    self.Out = self.Activation.forward(self.Out)

    return self.Out.reshape(self.OutShape)

  def backward(self, OutError):
    OutError = OutError.flatten()
    
    OutGradient = self.Activation.backward(self.Out) * OutError
      
    InputError = self.Weights @ OutError
      
    self.Biases += OutGradient * self.LearningRate
    self.Weights += np.outer(self.In.T, OutGradient) * self.LearningRate
    return InputError.reshape(self.InShape)