import numpy as np
from testsrc.activation import *

class rnn:
  def __init__(self, InSize, OutSize, HidSize=64, LearningRate=0.05):
    self.LearningRate, self.InSize, self.HidSize, self.OutSize = LearningRate, InSize, HidSize, OutSize

    self.WeightsHidToHid = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsInToHid = np.random.uniform(-0.005, 0.005, (HidSize, InSize))
    self.WeightsHidToOut = np.random.uniform(-0.005, 0.005, (OutSize, HidSize))

    self.HidBiases = np.zeros(HidSize)
    self.OutBiases = np.zeros(OutSize)

    self.Out = np.zeros(OutSize)

  def forwardprop(self, In):
    self.In = In
    self.Hid = np.zeros((len(self.In) + 1, self.HidSize))
    self.Out = np.zeros((len(self.In), self.OutSize))

    for i in range(len(In)):
      self.Hid[i + 1, :] = ApplyActivation(self.WeightsInToHid @ In[i] + self.WeightsHidToHid @ self.Hid[i, :] + self.HidBiases, 'Tanh')
      self.Out[i, :] = ApplyActivation(self.WeightsHidToOut @ self.Hid[i + 1, :] + self.OutBiases, 'Identity')
      
    return self.Out

  def backprop(self, OutError):
    WeightsInToHidΔ = np.zeros(self.WeightsInToHid.shape)
    WeightsHidToHidΔ = np.zeros(self.WeightsHidToHid.shape)
    HidBiasesΔ = np.zeros(self.HidBiases.shape)

    WeightsHidToOutΔ = np.zeros(self.WeightsHidToOut.shape)
    OutBiasesΔ = np.zeros(self.OutBiases.shape)

    HidError = self.WeightsHidToOut.T @ OutError[len(self.In) - 1]

    for i in reversed(range(len(self.In))):
      HidGradient = np.multiply(ActivationDerivative(self.Hid[i + 1], 'Tanh'), HidError)
      OutGradient = np.multiply(ActivationDerivative(self.Out[i], 'Identity'), OutError[i])

      WeightsInToHidΔ += np.outer(HidGradient, self.In[i].T)
      WeightsHidToHidΔ += np.outer(HidGradient, self.Hid[i].T)
      HidBiasesΔ += HidGradient

      WeightsHidToOutΔ += np.outer(OutGradient, self.Hid[i].T)
      OutBiasesΔ += OutGradient

      HidError = self.WeightsHidToHid.T @ HidGradient + self.WeightsHidToOut.T @ OutError[i]

    self.WeightsInToHid += self.LearningRate * np.clip(WeightsInToHidΔ, -1, 1)
    self.WeightsHidToHid += self.LearningRate * np.clip(WeightsHidToHidΔ, -1, 1)
    self.HidBiases += self.LearningRate * np.clip(HidBiasesΔ, -1, 1)
    
    self.WeightsHidToOut += self.LearningRate * np.clip(WeightsHidToOutΔ, -1, 1)
    self.OutBiases += self.LearningRate * np.clip(OutBiasesΔ, -1, 1)