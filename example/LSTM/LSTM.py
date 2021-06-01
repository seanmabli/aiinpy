import numpy as np
from ActivationFunctions import Sigmoid, Tanh, ReLU, LeakyReLU, StableSoftMax
Sigmoid, Tanh, ReLU, LeakyReLU, StableSoftMax = Sigmoid(), Tanh(), ReLU(), LeakyReLU(), StableSoftMax()

class LSTM:
  def __init__(self, InSize, OutSize, HidSize=64, LearningRate=0.05):
    self.LearningRate, self.HidSize = LearningRate, HidSize

    self.WeightsHidToForgetGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToInputGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToOutputGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToCellMemGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))

    self.WeightsInToForgetGate = np.random.uniform(-0.005, 0.005, (HidSize, InSize))
    self.WeightsInToInGate = np.random.uniform(-0.005, 0.005, (HidSize, InSize))
    self.WeightsInToOutGate = np.random.uniform(-0.005, 0.005, (HidSize, InSize))
    self.WeightsInToCellMemGate = np.random.uniform(-0.005, 0.005, (HidSize, InSize))

    self.ForgetGateBiases = np.zeros(HidSize)
    self.InGateBiases = np.zeros(HidSize)
    self.OutGateBiases = np.zeros(HidSize)
    self.CellMemGateBiases = np.zeros(HidSize)

    self.WeightsHidToOut = np.random.uniform(-0.005, 0.005, (OutSize, HidSize))
    self.OutBias = np.zeros(OutSize)
  
  def ForwardProp(self, Input):
    self.InputLayer = InputLayer
    self.Hid = np.zeros((len(self.InputLayer) + 1, self.HidSize))
    self.CellMem = np.zeros((len(self.InputLayer) + 1, self.HidSize))

    for i in range(len(InputLayer)):
      self.ForgetGate = Sigmoid((self.WeightsInToForgetGate @ self.InputLayer) + (self.WeightsHidToForgetGate @ self.Hid[i, :]) + self.ForgetGateBiases)
      self.InGate = Sigmoid((self.WeightsInToInGate @ self.InputLayer) + (self.WeightsHidToInputGate @ self.Hid[i, :]) + self.InGateBiases)
      self.OutGate = Sigmoid((self.WeightsInToOutGate @ self.InputLayer) + (self.WeightsHidToOutputGate @ self.Hid[i, :]) + self.OutGateBiases)
      self.CellMemGate = Tanh((self.WeightsInToCellMemGate @ self.InputLayer) + (self.WeightsHidToCellMemGate @ self.Hid[i, :]) + self.CellMemGateBiases)

      self.CellMem[i + 1, :] = (self.ForgetGate * self.CellMem[i, :]) + (self.InGate * self.CellMemGate)
      self.Hid[i + 1, :] = self.OutGate * Tanh(self.CellMem[i + 1, :])


  def BackProp(self):
    print("hi")