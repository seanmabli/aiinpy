import numpy as np
from .Activation import *
import sys

class LSTM:
  def __init__(self, InSize, OutSize, HidSize=64, LearningRate=0.05):
    self.LearningRate, self.HidSize, self.OutSize = LearningRate, HidSize, OutSize

    self.WeightsInToForgetGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))
    self.WeightsInToInGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))
    self.WeightsInToOutGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))
    self.WeightsInToCellMemGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))

    self.WeightsHidToForgetGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToInputGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToOutputGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToCellMemGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))

    self.ForgetGateBiases = np.zeros(HidSize)
    self.InGateBiases = np.zeros(HidSize)
    self.OutGateBiases = np.zeros(HidSize)
    self.CellMemGateBiases = np.zeros(HidSize)

    self.WeightsHidToOut = np.random.uniform(-0.005, 0.005, (HidSize, OutSize))
    self.OutBias = np.zeros(OutSize)
  
  def ForwardProp(self, In):
    self.In = In
    self.CellSize = len(In)

    self.Hid = np.zeros((self.CellSize + 1, self.HidSize))
    self.CellMem = np.zeros((self.CellSize + 1, self.HidSize))
    self.Out = np.zeros((self.CellSize, self.OutSize))

    self.ForgetGate = np.zeros((self.CellSize, self.HidSize))
    self.InGate = np.zeros((self.CellSize, self.HidSize))
    self.OutGate = np.zeros((self.CellSize, self.HidSize))
    self.CellMemGate = np.zeros((self.CellSize, self.HidSize))

    for i in range(self.CellSize):
      self.ForgetGate[i, :] = ApplyActivation((self.WeightsInToForgetGate.T @ self.In[i, :]) + (self.WeightsHidToForgetGate.T @ self.Hid[i, :]) + self.ForgetGateBiases, 'Sigmoid')
      self.InGate[i, :] = ApplyActivation((self.WeightsInToInGate.T @ self.In[i, :]) + (self.WeightsHidToInputGate.T @ self.Hid[i, :]) + self.InGateBiases, 'Sigmoid')
      self.OutGate[i, :] = ApplyActivation((self.WeightsInToOutGate.T @ self.In[i, :]) + (self.WeightsHidToOutputGate.T @ self.Hid[i, :]) + self.OutGateBiases, 'Sigmoid')
      self.CellMemGate[i, :] = ApplyActivation((self.WeightsInToCellMemGate.T @ self.In[i, :]) + (self.WeightsHidToCellMemGate.T @ self.Hid[i, :]) + self.CellMemGateBiases, 'Tanh')

      self.CellMem[i + 1, :] = (self.ForgetGate[i, :] * self.CellMem[i, :]) + (self.InGate[i, :] * self.CellMemGate[i, :])
      self.Hid[i + 1, :] = self.OutGate[i, :] * ApplyActivation(self.CellMem[i + 1, :], 'Tanh')
      self.Out[i, :] = self.WeightsHidToOut.T @ self.Hid[i + 1, :] + self.OutBias

    return self.Out

  def BackProp(self, OutError):
    InError = np.zeros(self.In.shape)
    HidError = np.zeros(self.HidSize)
    CellMemError = np.zeros(self.HidSize)

    ForgetGateError = np.zeros(self.HidSize)
    InGateError = np.zeros(self.HidSize)
    OutGateError = np.zeros(self.HidSize)
    CellMemGateError = np.zeros(self.HidSize)

    for i in reversed(range(self.CellSize)):
      OutGradient = ActivationDerivative(self.Out[i, :], 'StableSoftmax') * OutError[i, :]
      
      # CellMemError += HidError * ActivationDerivative(self.CellMem[i + 1, :], 'Tanh') * self.OutGate[i]
      # HidError += self.WeightsHidToOut @ OutError[i, :]

      # ForgetGateError = CellMemError * self.CellMem[i, :]
      # InGateError = CellMemError * self.CellMemGate[i, :]
      # OutGateError = HidError * ApplyActivation(self.CellMem[i + 1, :], 'Tanh')
      # CellMemGateError = CellMemError * self.InGate[i, :]

      # ForgetGateGradient = ActivationDerivative(self.ForgetGate[i, :], 'Sigmoid') * ForgetGateError
      # InGateGradient = ActivationDerivative(self.InGate[i, :], 'Sigmoid') * InGateError
      # OutGateGradient = ActivationDerivative(self.OutGate[i, :], 'Sigmoid') * OutGateError
      # CellMemGateGradient = ActivationDerivative(self.CellMemGate[i, :], 'Tanh') * CellMemGateError

      # CellMemError = CellMemError * self.ForgetGate[i, :]

      # HidError = self.WeightsHidToForgetGate @ ForgetGateError + self.WeightsHidToInputGate @ InGateError + self.WeightsHidToOutputGate @ OutGateError + self.WeightsHidToCellMemGate @ CellMemGateError
      # InError[i, :] = self.WeightsInToForgetGate @ ForgetGateError + self.WeightsInToInGate @ InGateError + self.WeightsInToOutGate @ OutGateError + self.WeightsInToCellMemGate @ CellMemGateError

      # self.WeightsInToForgetGate += np.outer(self.In[i, :].T, ForgetGateGradient) * self.LearningRate
      # self.WeightsInToInGate += np.outer(self.In[i, :].T, InGateGradient) * self.LearningRate
      # self.WeightsInToOutGate += np.outer(self.In[i, :].T, OutGateGradient) * self.LearningRate
      # self.WeightsInToCellMemGate += np.outer(self.In[i, :].T, CellMemGateGradient) * self.LearningRate

      # self.WeightsHidToForgetGate += np.outer(self.Hid[i, :].T, ForgetGateGradient) * self.LearningRate
      # self.WeightsHidToInputGate += np.outer(self.Hid[i, :].T, InGateGradient) * self.LearningRate
      # self.WeightsHidToOutputGate += np.outer(self.Hid[i, :].T, OutGateGradient) * self.LearningRate
      # self.WeightsHidToCellMemGate += np.outer(self.Hid[i, :].T, CellMemGateGradient) * self.LearningRate

      # self.ForgetGateBiases += ForgetGateGradient * self.LearningRate
      # self.InGateBiases += InGateGradient * self.LearningRate
      # self.OutGateBiases += OutGateGradient * self.LearningRate
      # self.CellMemGateBiases += CellMemGateGradient * self.LearningRate

      self.WeightsHidToOut += np.outer(self.Hid[i].T, OutGradient) * self.LearningRate
      self.OutBias += OutGradient * self.LearningRate

    return InError