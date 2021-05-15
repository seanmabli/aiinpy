import numpy as np
from .ActivationFunctions import Sigmoid, DerivativeOfSigmoid
from .ActivationFunctions import Tanh, DerivativeOfTanh
from .ActivationFunctions import ReLU, DerivativeOfReLU
from .ActivationFunctions import LeakyReLU, DerivativeOfLeakyReLU
from .ActivationFunctions import StableSoftMax, DerivativeOfStableSoftMax

class RNN:
  def __init__(self, InputSize, OutputSize, Type, HiddenSize=64, LearningRate=0.05):
    self.LearningRate = LearningRate
    self.Type = Type

    self.WeightsHidToHid = np.random.randn(HiddenSize, HiddenSize) / 1000
    self.WeightsInputToHid = np.random.randn(HiddenSize, InputSize) / 1000
    self.WeightsHidToOut = np.random.randn(OutputSize, HiddenSize) / 1000

    self.HiddenBiases = np.zeros(HiddenSize)
    self.OutputBiases = np.zeros(OutputSize)

  def ForwardProp(self, InputLayer):
    self.InputLayer = InputLayer
    self.Hidden = np.zeros((len(self.InputLayer) + 1, self.WeightsHidToHid.shape[0]))

    for i in range(len(InputLayer)):
      self.Hidden[i + 1, :] = Tanh(self.WeightsInputToHid @ InputLayer[i] + self.WeightsHidToHid @ self.Hidden[i, :] + self.HiddenBiases)
    
    self.Output = StableSoftMax(self.WeightsHidToOut @ self.Hidden[len(InputLayer), :] + self.OutputBiases)
    return self.Output

  def BackProp(self, OutputError):
    OutputGradient = np.multiply(DerivativeOfStableSoftMax(self.Output), OutputError)
    
    self.WeightsHidToOutDeltas = np.outer(OutputGradient, np.transpose(self.Hidden[len(self.InputLayer)]))
    self.OutputBiasesDeltas = OutputGradient

    self.WeightsHidToHidDeltas = np.zeros(self.WeightsHidToHid.shape)
    self.WeightsInputToHidDeltas = np.zeros(self.WeightsInputToHid.shape)
    self.HiddenBiasesDeltas = np.zeros(self.HiddenBiases.shape)

    self.HiddenError = np.transpose(self.WeightsHidToOut) @ OutputError

    for i in reversed(range(len(self.InputLayer))):
      self.HiddenGradient = np.multiply(DerivativeOfTanh(self.Hidden[i + 1]), self.HiddenError)

      self.HiddenBiasesDeltas += self.HiddenGradient
      self.WeightsHidToHidDeltas += np.outer(self.HiddenGradient, np.transpose(self.Hidden[i]))
      self.WeightsInputToHidDeltas += np.outer(self.HiddenGradient, np.transpose(self.InputLayer[i]))

      self.HiddenError = np.transpose(self.WeightsHidToHid) @ self.HiddenGradient

    self.WeightsHidToHid += self.LearningRate * np.clip(self.WeightsHidToHidDeltas, -1, 1)
    self.WeightsInputToHid += self.LearningRate * np.clip(self.WeightsInputToHidDeltas, -1, 1)
    self.WeightsHidToOut += self.LearningRate * np.clip(self.WeightsHidToOutDeltas, -1, 1)
    self.HiddenBiases += self.LearningRate * np.clip(self.HiddenBiasesDeltas, -1, 1)
    self.OutputBiases += self.LearningRate * np.clip(self.OutputBiasesDeltas, -1, 1)