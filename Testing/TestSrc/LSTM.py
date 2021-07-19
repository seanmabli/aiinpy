import numpy as np
from .ActivationFunctions import ForwardProp, BackProp

class LSTM:
  def __init__(self, InSize, OutSize, HidSize=64, LearningRate=0.05):
    self.LearningRate, self.HidSize, self.OutSize = LearningRate, HidSize, OutSize

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
    self.CellSize = len(InputLayer)

    self.Hid = np.zeros((self.CellSize + 1, self.HidSize))
    self.CellMem = np.zeros((self.CellSize + 1, self.HidSize))
    self.Out = np.zeros((self.CellSize, self.OutSize))

    self.ForgetGate = np.zeros((self.CellSize, self.HidSize))
    self.InGate = np.zeros((self.CellSize, self.HidSize))
    self.OutGate = np.zeros((self.CellSize, self.HidSize))
    self.CellMemGate = np.zeros((self.CellSize, self.HidSize))

    for i in range(self.CellSize):
      self.ForgetGate[i, :] = Sigmoid((self.WeightsInToForgetGate @ self.InputLayer) + (self.WeightsHidToForgetGate @ self.Hid[i, :]) + self.ForgetGateBiases)
      self.InGate[i, :] = Sigmoid((self.WeightsInToInGate @ self.InputLayer) + (self.WeightsHidToInputGate @ self.Hid[i, :]) + self.InGateBiases)
      self.OutGate[i, :] = Sigmoid((self.WeightsInToOutGate @ self.InputLayer) + (self.WeightsHidToOutputGate @ self.Hid[i, :]) + self.OutGateBiases)
      self.CellMemGate[i, :] = Tanh((self.WeightsInToCellMemGate @ self.InputLayer) + (self.WeightsHidToCellMemGate @ self.Hid[i, :]) + self.CellMemGateBiases)

      self.CellMem[i + 1, :] = (self.ForgetGate * self.CellMem[i, :]) + (self.InGate * self.CellMemGate)
      self.Hid[i + 1, :] = self.OutGate * Tanh(self.CellMem[i + 1, :])
      self.Out[i, :] = StableSoftMax((self.WeightsHidToOut @ self.Hid[i + 1, :]) + self.OutBias)
      
    return self.Out

  def BackProp(self, OutError):
    self.ForgetGateError = np.zeros(self.ForgetGate.shape)
    self.InGateError = np.zeros(self.InGate.shape)
    self.OutGateError = np.zeros(self.OutGate.shape)
    self.CellMemGateError = np.zeros(self.CellMemGate.shape)

    self.HidError = np.zeros(self.Hid.shape)
    self.CellMemError = np.zeros(self.CellMem.shape)

    for i in reversed(range(self.CellSize)):
      OutGradient = np.multiply(StableSoftMax.Derivative(self.Output), OutputError)
  
      self.HidError[i] = self.WeightsHidToOut @ OutError
  
      self.WeightsHidToOut += np.outer(OutGradient, np.transpose(self.Hidden[self.CellSize - 1])) * self.LearningRate
      self.OutBias += OutGradient * self.LearningRate
      


'''
  Errors:
  x - HidOut
  - HidIn
  x - CellMemOut
  x - CellMemIn
  - ForgetGate
  - InGate
  - OutGate
  - CellMemGate
  - In
'''

'''
    self.HidOutError = np.transpose(self.WeightsHidToOut) @ OutputError
    self.HidInError = 
    self.CellMemOutError = self.HidOutError * self.Out * Tanh.Derivative(self.CellMem[len(self.InputLayer)])
    self.CellMemInError = self.CellMemOutError * self.ForgetGate

    self.ForgetGateError = self.CellMemOutError * self.CellMem[self.CellSize - 1]
    self.InGateError = self.CellMemOutError * self.CellMemGate[self.CellSize - 1]
    self.OutGateError = 
    self.CellMemGateError =
''' 

'''
# With Deltas
  def BackProp(self, OutError):
    OutGradient = np.multiply(StableSoftMax.Derivative(self.Output), OutputError)
    
    self.WeightsHidToOutDeltas = np.outer(OutGradient, np.transpose(self.Hid[len(self.InputLayer)]))
    self.OutBiasesDeltas = OutGradient

    self.WeightsHidToForgetGateDeltas = np.zeros(WeightsHidToForgetGate.shape)
    self.WeightsHidToInputGateDeltas = self.LearningRate
    self.WeightsHidToOutputGateDeltas = self.LearningRate
    self.WeightsHidToCellMemGateDeltas = self.LearningRate

    self.WeightsInToForgetGateDeltas = self.LearningRate
    self.WeightsInToInGateDeltas = self.LearningRate
    self.WeightsInToOutGateDeltas = self.LearningRate
    self.WeightsInToCellMemGateDeltas = self.LearningRate

    self.ForgetGateBiasesDeltas = self.LearningRate
    self.InGateBiasesDeltas = self.LearningRate
    self.OutGateBiasesDeltas  = self.LearningRate
    self.CellMemGateBiasesDeltas = self.LearningRate

    self.HiddenError = np.transpose(self.WeightsHidToOut) @ OutputError

    for i in reversed(range(len(self.InputLayer))):
      self.HiddenGradient = np.multiply(Tanh.Derivative(self.Hidden[i + 1]), self.HiddenError)

      self.HiddenBiasesDeltas += self.HiddenGradient
      self.WeightsHidToHidDeltas += np.outer(self.HiddenGradient, np.transpose(self.Hidden[i]))
      self.WeightsInputToHidDeltas += np.outer(self.HiddenGradient, np.transpose(self.InputLayer[i]))

      self.HiddenError = np.transpose(self.WeightsHidToHid) @ self.HiddenGradient

    self.WeightsHidToForgetGate = self.LearningRate
    self.WeightsHidToInputGate = self.LearningRate
    self.WeightsHidToOutputGate = self.LearningRate
    self.WeightsHidToCellMemGate = self.LearningRate

    self.WeightsInToForgetGate = self.LearningRate
    self.WeightsInToInGate = self.LearningRate
    self.WeightsInToOutGate = self.LearningRate
    self.WeightsInToCellMemGate = self.LearningRate

    self.ForgetGateBiases = self.LearningRate
    self.InGateBiases = self.LearningRate
    self.OutGateBiases = self.LearningRate
    self.CellMemGateBiases = self.LearningRate

    self.WeightsHidToOut += self.LearningRate * self.WeightsHidToOutDeltas
    self.OutBias += self.LearningRate * self.OutBiasesDeltas
    '''