import numpy as np
from .activationfunctions import Sigmoid, DerivativeOfSigmoid, StableSoftMax, DerivativeOfStableSoftMax, ReLU, DerivativeOfReLU, Tanh, DerivativeOfTanh

class NN:
  def __init__(self, CurrentLayerShape, FollowingLayerShape, Activation, LearningRate, WeightsInit=(-1, 1)):
    self.Weights = np.random.uniform(WeightsInit[0], WeightsInit[1], (CurrentLayerShape, FollowingLayerShape))
    self.Biases = np.zeros(FollowingLayerShape)
    self.Activation = Activation
    self.LearningRate = LearningRate
  
  def ForwardProp(self, InputLayer):
    self.InputLayer = InputLayer
    self.Output = np.transpose(self.Weights) @ InputLayer + self.Biases
  
    # Apply Activation Function
    if(self.Activation == "ReLU"):
      self.Output = ReLU(self.Output)
    if(self.Activation == "Sigmoid"):
      self.Output = Sigmoid(self.Output)
    if(self.Activation == "StableSoftMax"):
      self.Output = StableSoftMax(self.Output)
    return self.Output

  def BackProp(self, FollowingLayerError):
    # Calculate Gradients
    FollowingLayerGradient = np.zeros(self.Output.shape)
    if(self.Activation == "ReLU"):
      FollowingLayerGradient = np.multiply(DerivativeOfReLU(self.Output), FollowingLayerError) * self.LearningRate
    if(self.Activation == "Sigmoid"):
      FollowingLayerGradient = np.multiply(DerivativeOfSigmoid(self.Output), FollowingLayerError) * self.LearningRate
    if(self.Activation == "StableSoftMax"):
      FollowingLayerGradient = np.multiply(DerivativeOfStableSoftMax(self.Output), FollowingLayerError) * self.LearningRate
      
    # Calculate Current Layer Error
    CurrentLayerError = self.Weights @ FollowingLayerError
      
    # Apply Deltas To The Weights And Biases
    self.Biases += FollowingLayerGradient
    self.Weights += np.outer(np.transpose(self.InputLayer), FollowingLayerGradient)
    return CurrentLayerError