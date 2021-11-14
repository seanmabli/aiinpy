import numpy as np
from .activation import *

class gru:
  def __init__(self, InSize, OutSize, OutActivation, HidSize=64, LearningRate=0.05):
    self.InSize, self.OutSize, self.OutActivation, self.HidSize, self.LearningRate = InSize, OutSize, OutActivation, HidSize, LearningRate

    self.WeightsInToResetGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))
    self.WeightsInToUpdateGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))
    self.WeightsInToHidGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))

    self.WeightsHidToResetGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToUpdateGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToHidGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))

    self.HidGateBias = np.zeros(HidSize)
    self.ResetGateBias = np.zeros(HidSize)
    self.UpdateGateBias = np.zeros(HidSize)

    self.WeightsHidToOut = np.random.uniform(-0.005, 0.005, (HidSize, OutSize))
    self.OutBias = np.zeros(OutSize)

  def __copy__(self):
    return type(self)(self.InSize, self.OutSize, self.OutActivation, self.HidSize, self.LearningRate)

  def forward(self, In):
    self.In = In
    self.CellSize = len(In)

    self.Hid = np.zeros((self.CellSize + 1, self.HidSize))
    self.Out = np.zeros((self.CellSize, self.OutSize))
  
    self.ResetGate = np.zeros((self.CellSize, self.HidSize))
    self.UpdateGate = np.zeros((self.CellSize, self.HidSize))
    self.HidGate = np.zeros((self.CellSize, self.HidSize))

    for i in range(self.CellSize):
      self.ResetGate[i, :] = ApplyActivation(self.WeightsInToResetGate.T @ self.In[i, :] + self.WeightsHidToResetGate.T @ self.Hid[i, :] + self.ResetGateBias, 'Sigmoid')
      self.UpdateGate[i, :] = ApplyActivation(self.WeightsInToUpdateGate.T @ self.In[i, :] + self.WeightsHidToUpdateGate.T @ self.Hid[i, :] + self.UpdateGateBias, 'Sigmoid')
      self.HidGate[i, :] = ApplyActivation(self.WeightsInToHidGate.T @ self.In[i, :] + self.WeightsHidToHidGate.T @ (self.Hid[i, :] * self.ResetGate[i, :]) + self.HidGateBias, 'Tanh')
  
      self.Hid[i + 1, :] = (1 - self.UpdateGate[i, :]) * self.Hid[i, :] + self.UpdateGate[i, :] * self.HidGate[i, :]
      self.Out[i, :] = ApplyActivation(self.WeightsHidToOut.T @ self.Hid[i + 1, :] + self.OutBias, self.OutActivation)

    return self.Out

  def backprop(self, OutError):
    InError = np.zeros(self.In.shape)
    HidError = np.zeros(self.HidSize)

    WeightsInToResetGateΔ = np.zeros(self.WeightsInToResetGate.shape)
    WeightsInToUpdateGateΔ = np.zeros(self.WeightsInToUpdateGate.shape)
    WeightsInToHidGateΔ = np.zeros(self.WeightsInToHidGate.shape)

    WeightsHidToResetGateΔ = np.zeros(self.WeightsHidToResetGate.shape)
    WeightsHidToUpdateGateΔ = np.zeros(self.WeightsHidToUpdateGate.shape)
    WeightsHidToHidGateΔ = np.zeros(self.WeightsHidToHidGate.shape)

    HidGateBiasΔ = np.zeros(self.HidGateBias.shape)
    ResetGateBiasΔ = np.zeros(self.ResetGateBias.shape)
    UpdateGateBiasΔ = np.zeros(self.UpdateGateBias.shape)

    WeightsHidToOutΔ = np.zeros(self.WeightsHidToOut.shape)
    OutBiasΔ = np.zeros(self.OutBias.shape)

    for i in reversed(range(self.CellSize)):
      OutGradient = ActivationDerivative(self.Out[i, :], self.OutActivation) * OutError[i, :]

      HidError += self.WeightsHidToOut @ OutError[i, :]

      HidGateError = HidError * self.UpdateGate[i, :]
      UpdateGateError = HidError * (-1 * self.Hid[i, :]) + HidError * self.HidGate[i, :]
      ResetGateError = (self.WeightsHidToHidGate.T @ HidGateError) * self.Hid[i, :]

      HidError += (self.WeightsHidToHidGate.T @ HidGateError) * self.ResetGate[i, :] + self.WeightsHidToResetGate.T @ ResetGateError + self.WeightsHidToUpdateGate.T @ UpdateGateError
      InError[i, :] = self.WeightsInToResetGate @ ResetGateError + self.WeightsInToUpdateGate @ UpdateGateError + self.WeightsInToHidGate @ HidGateError

      ResetGateGradient = ActivationDerivative(self.ResetGate[i, :], 'Sigmoid') * ResetGateError
      UpdateGateGradient = ActivationDerivative(self.UpdateGate[i, :], 'Sigmoid') * UpdateGateError
      HidGateGradient = ActivationDerivative(self.HidGate[i, :], 'Tanh') * HidGateError

      WeightsInToResetGateΔ += np.outer(self.In[i, :].T, ResetGateGradient)
      WeightsInToUpdateGateΔ += np.outer(self.In[i, :].T, UpdateGateGradient)
      WeightsInToHidGateΔ += np.outer(self.In[i, :].T, HidGateGradient)
  
      WeightsHidToResetGateΔ += np.outer(self.Hid[i, :].T, ResetGateGradient)
      WeightsHidToUpdateGateΔ += np.outer(self.Hid[i, :].T, UpdateGateGradient)
      WeightsHidToHidGateΔ += np.outer(self.Hid[i, :].T, HidGateGradient)
  
      HidGateBiasΔ += ResetGateGradient
      ResetGateBiasΔ += UpdateGateGradient
      UpdateGateBiasΔ += HidGateGradient

      WeightsHidToOutΔ += np.outer(self.Hid[i].T, OutGradient)
      OutBiasΔ += OutGradient

    self.WeightsInToResetGate += np.clip(WeightsInToResetGateΔ, -1, 1) * self.LearningRate
    self.WeightsInToUpdateGate += np.clip(WeightsInToUpdateGateΔ, -1, 1) * self.LearningRate
    self.WeightsInToHidGate += np.clip(WeightsInToHidGateΔ, -1, 1) * self.LearningRate

    self.WeightsHidToResetGate += np.clip(WeightsHidToResetGateΔ, -1, 1) * self.LearningRate
    self.WeightsHidToUpdateGate += np.clip(WeightsHidToUpdateGateΔ, -1, 1) * self.LearningRate
    self.WeightsHidToHidGate += np.clip(WeightsHidToHidGateΔ, -1, 1) * self.LearningRate

    self.HidGateBias += np.clip(HidGateBiasΔ, -1, 1) * self.LearningRate
    self.ResetGateBias += np.clip(ResetGateBiasΔ, -1, 1) * self.LearningRate
    self.UpdateGateBias += np.clip(UpdateGateBiasΔ, -1, 1) * self.LearningRate

    self.WeightsHidToOut += np.clip(WeightsHidToOutΔ, -1, 1) * self.LearningRate
    self.OutBias += np.clip(OutBiasΔ, -1, 1) * self.LearningRate

    return InError