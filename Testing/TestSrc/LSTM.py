import numpy as np
from .Activation import ApplyActivation, ActivationDerivative

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
      self.ForgetGate[i, :] = ApplyActivation((self.WeightsInToForgetGate @ self.In[i, :]) + (self.WeightsHidToForgetGate @ self.Hid[i, :]) + self.ForgetGateBiases, 'Sigmoid')
      self.InGate[i, :] = ApplyActivation((self.WeightsInToInGate @ self.In[i, :]) + (self.WeightsHidToInputGate @ self.Hid[i, :]) + self.InGateBiases, 'Sigmoid')
      self.OutGate[i, :] = ApplyActivation((self.WeightsInToOutGate @ self.In[i, :]) + (self.WeightsHidToOutputGate @ self.Hid[i, :]) + self.OutGateBiases, 'Sigmoid')
      self.CellMemGate[i, :] = ApplyActivation((self.WeightsInToCellMemGate @ self.In[i, :]) + (self.WeightsHidToCellMemGate @ self.Hid[i, :]) + self.CellMemGateBiases, 'Tanh')

      self.CellMem[i + 1, :] = (self.ForgetGate[i, :] * self.CellMem[i, :]) + (self.InGate[i, :] * self.CellMemGate[i, :])
      self.Hid[i + 1, :] = self.OutGate[i, :] * ApplyActivation(self.CellMem[i + 1, :], 'Tanh')
      self.Out[i, :] = ApplyActivation((self.WeightsHidToOut @ self.Hid[i + 1, :]) + self.OutBias, 'StableSoftmax')
    
    return self.Out

  def BackProp(self, OutError):
    self.InError = np.zeros(self.In.shape)

    self.ForgetGateError = np.zeros(self.ForgetGate.shape)
    self.InGateError = np.zeros(self.InGate.shape)
    self.OutGateError = np.zeros(self.OutGate.shape)
    self.CellMemGateError = np.zeros(self.CellMemGate.shape)

    self.HidError = np.zeros(self.Hid.shape)
    self.CellMemError = np.zeros(self.CellMem.shape)

    OutGradient = np.multiply(ActivationDerivative(self.Out, 'StableSoftmax'), OutError)
    HidError = self.WeightsHidToOut @ OutError
  
    self.WeightsHidToOut += np.outer(OutGradient, np.transpose(self.Hidden)) * self.LearningRate
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