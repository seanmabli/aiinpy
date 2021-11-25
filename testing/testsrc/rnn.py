import numpy as np
from .activation import *

class rnn:
  def __init__(self, inshape, outshape, Type, outactivation=stablesoftmax(), HidSize=64, learningrate=0.05):
    self.learningrate, self.inshape, self.HidSize, self.outshape, self.Type, self.outactivation = learningrate, inshape, HidSize, outshape, Type, outactivation

    self.weightsHidToHid = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.weightsinToHid = np.random.uniform(-0.005, 0.005, (HidSize, inshape))
    self.weightsHidToout = np.random.uniform(-0.005, 0.005, (outshape, HidSize))

    self.Hidbiases = np.zeros(HidSize)
    self.outbiases = np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.inshape, self.outshape, self.Type, self.outactivation, self.HidSize, self.learningrate)

  def forward(self, input):
    self.input = input
    self.Hid = np.zeros((len(self.input) + 1, self.HidSize))
    
    if self.Type == 'ManyToOne':
      for i in range(len(input)):
        self.Hid[i + 1, :] = applyactivation(self.weightsinToHid @ input[i] + self.weightsHidToHid @ self.Hid[i, :] + self.Hidbiases, tanh())

      self.out = applyactivation(self.weightsHidToout @ self.Hid[len(input), :] + self.outbiases, self.outactivation)
    
    elif self.Type == 'ManyToMany':
      self.out = np.zeros((len(self.input), self.outshape))

      for i in range(len(input)):
        self.Hid[i + 1, :] = applyactivation(self.weightsinToHid @ input[i] + self.weightsHidToHid @ self.Hid[i, :] + self.Hidbiases, tanh())
        self.out[i, :] = applyactivation(self.weightsHidToout @ self.Hid[i + 1, :] + self.outbiases, self.outactivation)

    return self.out

  def backward(self, outError):
    weightsinToHidΔ = np.zeros(self.weightsinToHid.shape)
    weightsHidToHidΔ = np.zeros(self.weightsHidToHid.shape)
    HidbiasesΔ = np.zeros(self.Hidbiases.shape)

    if self.Type == 'ManyToOne':
      outGradient = np.multiply(activationDerivative(self.out, self.outactivation), outError)

      weightsHidTooutΔ = np.outer(outGradient, self.Hid[len(self.input)].T)
      outbiasesΔ = outGradient

      HidError = self.weightsHidToout.T @ outError

      for i in reversed(range(len(self.input))):
        HidGradient = np.multiply(activationDerivative(self.Hid[i + 1], tanh()), HidError)

        HidbiasesΔ += HidGradient
        weightsHidToHidΔ += np.outer(HidGradient, self.Hid[i].T)
        weightsinToHidΔ += np.outer(HidGradient, self.input[i].T)

        HidError = self.weightsHidToHid.T @ HidGradient

    elif self.Type == 'ManyToMany':
      weightsHidTooutΔ = np.zeros(self.weightsHidToout.shape)
      outbiasesΔ = np.zeros(self.outbiases.shape)

      HidError = self.weightsHidToout.T @ outError[len(self.input) - 1]

      for i in reversed(range(len(self.input))):
        HidGradient = np.multiply(activationDerivative(self.Hid[i + 1], tanh()), HidError)
        outGradient = np.multiply(activationDerivative(self.out[i], self.outactivation), outError[i])

        weightsinToHidΔ += np.outer(HidGradient, self.input[i].T)
        weightsHidToHidΔ += np.outer(HidGradient, self.Hid[i].T)
        HidbiasesΔ += HidGradient

        weightsHidTooutΔ += np.outer(outGradient, self.Hid[i].T)
        outbiasesΔ += outGradient

        HidError = self.weightsHidToHid.T @ HidGradient + self.weightsHidToout.T @ outError[i]

    self.weightsinToHid += self.learningrate * np.clip(weightsinToHidΔ, -1, 1)
    self.weightsHidToHid += self.learningrate * np.clip(weightsHidToHidΔ, -1, 1)
    self.Hidbiases += self.learningrate * np.clip(HidbiasesΔ, -1, 1)

    self.weightsHidToout += self.learningrate * np.clip(weightsHidTooutΔ, -1, 1)
    self.outbiases += self.learningrate * np.clip(outbiasesΔ, -1, 1)