import numpy as np
from ActivationFunctions import Sigmoid, Tanh, ReLU, LeakyReLU, StableSoftMax
Sigmoid, Tanh, ReLU, LeakyReLU, StableSoftMax = Sigmoid(), Tanh(), ReLU(), LeakyReLU(), StableSoftMax()

class LSTM:
  def __init__(self, InputSize, OutputSize, Type, HidSize=64, LearningRate=0.05):
    self.LearningRate, self.Type = LearningRate, Type

    self.WeightsHidToHid = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.WeightsInputToHid = np.random.uniform(-0.005, 0.005, (HidSize, InputSize))
    self.WeightsHidToOut = np.random.uniform(-0.005, 0.005, (OutputSize, HidSize))

    self.HiddenBiases = np.zeros(HidSize)
    self.OutputBiases = np.zeros(OutputSize)