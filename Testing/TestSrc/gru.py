import numpy as np
from .activation import *

class gru:
  def __init__(self, InSize, OutSize, OutActivation, HidSize=64, LearningRate=0.05):
    self.WeightsInToResetGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))
    self.WeightsInToUpdateGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))
    self.WeightsInToHidGate = np.random.uniform(-0.005, 0.005, (InSize, HidSize))

    self.WeightsHidToResetGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToUpdateGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToHidGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))

    self.HidGateBias = np.zeros(HidSize)
    self.ResetGateBias = np.zeros(HidSize)
    self.UpdateGateBias = np.zeros(HidSize)

    self.WeightsOutToHid = np.random.uniform(-0.005, 0.005, (HidSize, OutSize))
    self.OutGateBias = np.zeros(OutSize)

  def forwardprop(self, In):
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
