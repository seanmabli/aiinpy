import numpy as np
from .binarystep import binarystep
from .gaussian import gaussian
from .identity import identity
from .leakyrelu import leakyrelu
from .mish import mish
from .relu import relu
from .selu import selu
from .sigmoid import sigmoid
from .silu import silu
from .softmax import softmax
from .softplus import softplus
from .stablesoftmax import stablesoftmax
from .tanh import tanh

class rnn:
  def __init__(self, outshape, type, outactivation=stablesoftmax(), hidshape=64, learningrate=0.05, inshape=None):
    self.learningrate, self.type, self.outactivation = learningrate, type, outactivation
    self.inshape, self.hidshape, self.outshape = inshape, hidshape, outshape
    
    if inshape is not None:
      self.weightsinToHid = np.random.uniform(-0.005, 0.005, (np.prod(hidshape), np.prod(inshape)))

    self.weightsHidToHid = np.random.uniform(-0.005, 0.005, (np.prod(hidshape), np.prod(hidshape)))
    self.Hidbiases = np.zeros(hidshape)

    self.weightsHidToout = np.random.uniform(-0.005, 0.005, (np.prod(outshape), np.prod(hidshape)))
    self.outbiases = np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.outshape, self.type, self.outactivation, self.hidshape, self.learningrate, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape
    self.weightsinToHid = np.random.uniform(-0.005, 0.005, (np.prod(self.hidshape), np.prod(inshape)))
    return self.outshape

  def forward(self, input):
    self.input = input
    self.Hid = np.zeros((len(self.input) + 1, self.hidshape))
    
    if self.type == 'ManyToOne':
      for i in range(len(input)):
        self.Hid[i + 1, :] = tanh().forward(self.weightsinToHid @ input[i].flatten() + self.weightsHidToHid @ self.Hid[i, :] + self.Hidbiases)

      self.out = self.outactivation.forward(self.weightsHidToout @ self.Hid[len(input), :] + self.outbiases)
    
    elif self.type == 'ManyToMany':
      self.out = np.zeros((len(self.input), self.outshape))

      for i in range(len(input)):
        self.Hid[i + 1, :] = tanh().forward(self.weightsinToHid @ input[i].flatten() + self.weightsHidToHid @ self.Hid[i, :] + self.Hidbiases)
        self.out[i, :] = self.outactivation.forward(self.weightsHidToout @ self.Hid[i + 1, :] + self.outbiases)

    return self.out

  def backward(self, outError):
    weightsinToHid?? = np.zeros(self.weightsinToHid.shape)
    weightsHidToHid?? = np.zeros(self.weightsHidToHid.shape)
    Hidbiases?? = np.zeros(self.Hidbiases.shape)

    if self.type == 'ManyToOne':
      outGradient = np.multiply(self.outactivation.backward(self.out), outError)

      weightsHidToout?? = np.outer(outGradient, self.Hid[len(self.input)].T)
      outbiases?? = outGradient

      HidError = self.weightsHidToout.T @ outError

      for i in reversed(range(len(self.input))):
        HidGradient = np.multiply(tanh().backward(self.Hid[i + 1]), HidError)

        Hidbiases?? += HidGradient
        weightsHidToHid?? += np.outer(HidGradient, self.Hid[i].T)
        weightsinToHid?? += np.outer(HidGradient, self.input[i].T)

        HidError = self.weightsHidToHid.T @ HidGradient

    elif self.type == 'ManyToMany':
      weightsHidToout?? = np.zeros(self.weightsHidToout.shape)
      outbiases?? = np.zeros(self.outbiases.shape)

      HidError = self.weightsHidToout.T @ outError[len(self.input) - 1]

      for i in reversed(range(len(self.input))):
        HidGradient = np.multiply(tanh().backward(self.Hid[i + 1]), HidError)
        outGradient = np.multiply(self.outactivation.backward(self.out[i]), outError[i])

        weightsinToHid?? += np.outer(HidGradient, self.input[i].T)
        weightsHidToHid?? += np.outer(HidGradient, self.Hid[i].T)
        Hidbiases?? += HidGradient

        weightsHidToout?? += np.outer(outGradient, self.Hid[i].T)
        outbiases?? += outGradient

        HidError = self.weightsHidToHid.T @ HidGradient + self.weightsHidToout.T @ outError[i]

    self.weightsinToHid += self.learningrate * np.clip(weightsinToHid??, -1, 1)
    self.weightsHidToHid += self.learningrate * np.clip(weightsHidToHid??, -1, 1)
    self.Hidbiases += self.learningrate * np.clip(Hidbiases??, -1, 1)

    self.weightsHidToout += self.learningrate * np.clip(weightsHidToout??, -1, 1)
    self.outbiases += self.learningrate * np.clip(outbiases??, -1, 1)