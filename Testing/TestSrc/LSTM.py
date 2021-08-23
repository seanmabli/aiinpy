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

    self.ForgetGateError = np.zeros(self.HidSize)
    self.InGateError = np.zeros(self.HidSize)
    self.OutGateError = np.zeros(self.HidSize)
    self.CellMemGateError = np.zeros(self.HidSize)

    self.HidError = np.zeros(self.HidSize)
    self.CellMemError = np.zeros(self.HidSize)

    OutGradient = np.multiply(ActivationDerivative(self.Out, 'StableSoftmax'), OutError)
    HidError = self.WeightsHidToOut @ OutError
  
    self.WeightsHidToOutΔ = np.outer(OutGradient, np.transpose(self.Hidden))
    self.OutBiasΔ = OutGradient
    
    '''
    for i in reversed(range(self.CellSize)):

      

      ForgetGateGradient = np.multiply(ActivationDerivative(self.ForgetGate[i, :], 'Sigmoid'), self.ForgetGateError[i, :])
      InGateGradient = np.multiply(ActivationDerivative(self.InGate[i, :], 'Sigmoid'), self.InGateError[i, :])
      OutGateGradient = np.multiply(ActivationDerivative(self.OutGate[i, :], 'Sigmoid'), self.OutGateError[i, :])
      CellMemGateGradient = np.multiply(ActivationDerivative(self.CellMemGate[i, :], 'Tanh'), self.CellMemGateError[i, :])

      self.WeightsInToForgetGateΔ = np.outer(ForgetGateGradient, np.transpose(self.In[i, :]))
      self.WeightsHidToForgetGateΔ = np.outer(ForgetGateGradient, np.transpose(self.Hid[i, :]))
      self.ForgetGateBiasesΔ = ForgetGateGradient

      self.WeightsInToInGateΔ = np.outer(InGateGradient, np.transpose(self.In[i, :]))
      self.WeightsHidToInputGateΔ = np.outer(InGateGradient, np.transpose(self.Hid[i, :]))
      self.InGateBiasesΔ = InGateGradient

      self.WeightsInToOutGateΔ = np.outer(OutGateGradient, np.transpose(self.In[i, :]))
      self.WeightsHidToOutputGateΔ = np.outer(OutGateGradient, np.transpose(self.Hid[i, :]))
      self.OutGateBiasesΔ = OutGateGradient

      self.WeightsInToCellMemGateΔ = np.outer(CellMemGateGradient, np.transpose(self.In[i, :]))
      self.WeightsHidToCellMemGateΔ = np.outer(CellMemGateGradient, np.transpose(self.Hid[i, :]))
      self.CellMemGateBiasesΔ = CellMemGateGradient

      self.HidError = self.WeightsHidToForgetGate @ ForgetGateError + self.WeightsHidToInputGate @ InGateError + self.WeightsHidToOutputGate @ OutGateError + self.WeightsHidToCellMemGate @ CellMemGateError
      self.InError[i, :] = self.WeightsInToForgetGate @ ForgetGateError + self.WeightsInToInGate @ InGateError + self.WeightsInToOutGate @ OutGateError + self.WeightsInToCellMemGate @ CellMemGateError

    return self.InError
    '''