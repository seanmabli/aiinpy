import numpy as np
from .ActivationFunctions import ForwardProp, BackProp

class NN:
  def __init__(self, InputSize, OutputSize, Activation, LearningRate, WeightsInit=(-1, 1), BiasesInit=(0, 0), DropoutRate=0):
    self.Weights = np.random.uniform(WeightsInit[0], WeightsInit[1], (InputSize, OutputSize))
    self.Biases = np.random.uniform(BiasesInit[0], BiasesInit[1], (OutputSize))
    self.Activation, self.LearningRate, self.DropoutRate = Activation, LearningRate, DropoutRate
  
  def ForwardProp(self, InputLayer):
    self.InputLayer = InputLayer
    self.Output = np.transpose(self.Weights) @ InputLayer + self.Biases
  
    # Apply Activation Function
    self.Output = ForwardProp(self.Output, self.Activation)

    self.Dropout = np.random.binomial(1, self.DropoutRate, size=len(self.Output))
    self.Dropout = np.where(self.Dropout == 0, 1, 0)
    self.Output *= self.Dropout

    return self.Output

  def BackProp(self, FollowingLayerError):
    FollowingLayerError *= self.Dropout

    # Apply Activation Function Derivative
    FollowingLayerGradient = np.multiply(BackProp(self.Output, self.Activation), FollowingLayerError)
      
    # Calculate Current Layer Error
    CurrentLayerError = self.Weights @ FollowingLayerError
      
    # Apply Deltas To The Weights And Biases
    self.Biases += FollowingLayerGradient * self.LearningRate
    self.Weights += np.outer(np.transpose(self.InputLayer), FollowingLayerGradient) * self.LearningRate
    return CurrentLayerError