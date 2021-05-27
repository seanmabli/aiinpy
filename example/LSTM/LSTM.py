import numpy as np
from ActivationFunctions import Sigmoid, Tanh, ReLU, LeakyReLU, StableSoftMax
Sigmoid, Tanh, ReLU, LeakyReLU, StableSoftMax = Sigmoid(), Tanh(), ReLU(), LeakyReLU(), StableSoftMax()

class LSTM:
  def __init__(self, InSize, OutSize, HidSize=64, LearningRate=0.05):
    self.LearningRate = LearningRate

    self.WeightsHidToInputGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToOutputGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToCellMemGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsHidToForgetGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))

    self.WeightsInToInGate = np.random.uniform(-0.005, 0.005, (HidSize, InSize))
    self.WeightsInToOutGate = np.random.uniform(-0.005, 0.005, (HidSize, InSize))
    self.WeightsInToCellMemGate = np.random.uniform(-0.005, 0.005, (HidSize, InSize))
    self.WeightsInToForgetGate = np.random.uniform(-0.005, 0.005, (HidSize, InSize))

    self.InGateBiases = np.zeros(HidSize)
    self.OuGateBiases = np.zeros(HidSize)
    self.CellMemGateBiases = np.zeros(HidSize)
    self.ForgetGateBiases = np.zeros(HidSize)

    self.WeightsHidToOut = np.random.uniform(-0.005, 0.005, (OutSize, HidSize))
    self.OutBias = np.zeros(OutSize)
  
  def ForwardProp(self, Input):
    print("hi")

  def BackProp(self):
    print("hi")