import numpy as np
from .Activation import *

class RNN:
  def __init__(self, InSize, OutSize, HidSize=64, LearningRate=0.05):
    self.LearningRate, self.HidSize = LearningRate, HidSize

    self.WeightsHidToHid = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsInputToHid = np.random.uniform(-0.005, 0.005, (HidSize, InSize))
    self.WeightsHidToOut = np.random.uniform(-0.005, 0.005, (OutSize, HidSize))

    self.HidBiases = np.zeros(HidSize)
    self.OutBiases = np.zeros(OutSize)

    self.Out = np.zeros(OutSize)

  def forwardprop(self, InputLayer):
    self.InputLayer = InputLayer
    self.Hid = np.zeros((len(self.InputLayer) + 1, self.HidSize))
    self.Out = np.zeros((len(self.InputLayer), self.OutSize))

    for i in range(len(InputLayer)):
      self.Hid[i + 1, :] = ApplyActivation(self.WeightsInputToHid @ InputLayer[i] + self.WeightsHidToHid @ self.Hidden[i, :] + self.HiddenBiases, 'Tanh')
      self.Out[i, :] = ApplyActivation(self.WeightsHidToOut @ self.Hid[i + 1, :] + self.OutputBiases, 'StableSoftmax')
      
    return self.Out

  def backprop(self, OutError):
    OutGradient = np.multiply(ActivationDerivative(self.Out, 'StableSoftmax'), OutError)
    
    self.WeightsHidToOutΔ = np.outer(OutGradient, self.Hid[len(self.InputLayer)].T)
    self.OutBiasesΔ = OutGradient

    self.WeightsHidToHidΔ = np.zeros(self.WeightsHidToHid.shape)
    self.WeightsInputToHidΔ = np.zeros(self.WeightsInputToHid.shape)
    self.HidBiasesΔ = np.zeros(self.HidBiases.shape)

    self.HidError = self.WeightsHidToOut.T @ OutError

    for i in reversed(range(len(self.InputLayer))):
      self.HidGradient = np.multiply(ActivationDerivative(self.Hid[i + 1], 'Tanh'), self.HidError)

      self.HidBiasesΔ += self.HidGradient
      self.WeightsHidToHidΔ += np.outer(self.HidGradient, self.Hid[i].T)
      self.WeightsInputToHidΔ += np.outer(self.HidGradient, self.InputLayer[i].T)

      self.HidError = self.WeightsHidToHid.T @ self.HidGradient

    self.WeightsHidToHid += self.LearningRate * np.clip(self.WeightsHidToHidΔ, -1, 1)
    self.WeightsInputToHid += self.LearningRate * np.clip(self.WeightsInputToHidΔ, -1, 1)
    self.WeightsHidToOut += self.LearningRate * np.clip(self.WeightsHidToOutΔ, -1, 1)
    self.HidBiases += self.LearningRate * np.clip(self.HidBiasesΔ, -1, 1)
    self.OutputBiases += self.LearningRate * np.clip(self.OutputBiasesΔ, -1, 1)