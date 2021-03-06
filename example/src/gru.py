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

class gru:
  def __init__(self, outshape, outactivation, hidshape=64, learningrate=0.05, inshape=None):
    self.outactivation, self.learningrate = outactivation, learningrate
    self.inshape, self.hidshape, self.outshape = inshape, hidshape, outshape

    if inshape is not None:
      self.weightsinToResetGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinToUpdateGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinTohidGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))

    self.weightshidToResetGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidToUpdateGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidTohidGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))

    self.hidGatebias = np.zeros(hidshape)
    self.ResetGatebias = np.zeros(hidshape)
    self.UpdateGatebias = np.zeros(hidshape)

    self.weightshidToout = np.random.uniform(-0.005, 0.005, (hidshape, outshape))
    self.outbias = np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.outshape, self.outactivation, self.hidshape, self.learningrate, self.inshape)

  def __repr__(self):
    return 'gru(inshape=' + str(self.inshape) + ', hidshape=' + str(self.hidshape) + ', outshape=' + str(self.outshape) + ', outactivation=' + str(self.outactivation) + ', learningrate=' + str(self.learningrate) + ')'

  def modelinit(self, inshape):
    self.inshape = inshape
    self.weightsinToResetGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinToUpdateGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinTohidGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    return self.outshape

  def forward(self, input):
    self.input = input
    self.cellSize = len(input)

    self.hid = np.zeros((self.cellSize + 1, self.hidshape))
    self.out = np.zeros((self.cellSize, self.outshape))
    self.outderivative = np.zeros((self.cellSize, self.outshape))
  
    self.ResetGate = np.zeros((self.cellSize, self.hidshape))
    self.UpdateGate = np.zeros((self.cellSize, self.hidshape))
    self.hidGate = np.zeros((self.cellSize, self.hidshape))

    self.resetgatederivative = np.zeros((self.cellSize, self.hidshape))
    self.updategatederivative = np.zeros((self.cellSize, self.hidshape))
    self.hidgatederivative = np.zeros((self.cellSize, self.hidshape))


    for i in range(self.cellSize):
      self.ResetGate[i, :] = sigmoid().forward(self.weightsinToResetGate.T @ self.input[i, :] + self.weightshidToResetGate.T @ self.hid[i, :] + self.ResetGatebias)
      self.resetgatederivative[i, :] = sigmoid().backward(self.weightsinToResetGate.T @ self.input[i, :] + self.weightshidToResetGate.T @ self.hid[i, :] + self.ResetGatebias)
      self.UpdateGate[i, :] = sigmoid().forward(self.weightsinToUpdateGate.T @ self.input[i, :] + self.weightshidToUpdateGate.T @ self.hid[i, :] + self.UpdateGatebias)
      self.updategatederivative[i, :] = sigmoid().backward(self.weightsinToUpdateGate.T @ self.input[i, :] + self.weightshidToUpdateGate.T @ self.hid[i, :] + self.UpdateGatebias)
      self.hidGate[i, :] = tanh().forward(self.weightsinTohidGate.T @ self.input[i, :] + self.weightshidTohidGate.T @ (self.hid[i, :] * self.ResetGate[i, :]) + self.hidGatebias)
      self.hidgatederivative[i, :] = tanh().backward(self.weightsinTohidGate.T @ self.input[i, :] + self.weightshidTohidGate.T @ (self.hid[i, :] * self.ResetGate[i, :]) + self.hidGatebias)

      self.hid[i + 1, :] = (1 - self.UpdateGate[i, :]) * self.hid[i, :] + self.UpdateGate[i, :] * self.hidGate[i, :]
      self.out[i, :] = self.outactivation.forward(self.weightshidToout.T @ self.hid[i + 1, :] + self.outbias)
      self.outderivative[i, :] = self.outactivation.backward(self.weightshidToout.T @ self.hid[i + 1, :] + self.outbias)
    
    return self.out

  def backward(self, outError):
    inError = np.zeros(self.input.shape)
    hidError = np.zeros(self.hidshape)

    weightsinToResetGate?? = np.zeros(self.weightsinToResetGate.shape)
    weightsinToUpdateGate?? = np.zeros(self.weightsinToUpdateGate.shape)
    weightsinTohidGate?? = np.zeros(self.weightsinTohidGate.shape)

    weightshidToResetGate?? = np.zeros(self.weightshidToResetGate.shape)
    weightshidToUpdateGate?? = np.zeros(self.weightshidToUpdateGate.shape)
    weightshidTohidGate?? = np.zeros(self.weightshidTohidGate.shape)

    hidGatebias?? = np.zeros(self.hidGatebias.shape)
    ResetGatebias?? = np.zeros(self.ResetGatebias.shape)
    UpdateGatebias?? = np.zeros(self.UpdateGatebias.shape)

    weightshidToout?? = np.zeros(self.weightshidToout.shape)
    outbias?? = np.zeros(self.outbias.shape)

    for i in reversed(range(self.cellSize)):
      outGradient = self.outderivative[i, :] * outError[i, :]

      hidError += self.weightshidToout @ outError[i, :]

      hidGateError = hidError * self.UpdateGate[i, :]
      UpdateGateError = hidError * (-1 * self.hid[i, :]) + hidError * self.hidGate[i, :]
      ResetGateError = (self.weightshidTohidGate.T @ hidGateError) * self.hid[i, :]

      hidError += (self.weightshidTohidGate.T @ hidGateError) * self.ResetGate[i, :] + self.weightshidToResetGate.T @ ResetGateError + self.weightshidToUpdateGate.T @ UpdateGateError
      inError[i, :] = self.weightsinToResetGate @ ResetGateError + self.weightsinToUpdateGate @ UpdateGateError + self.weightsinTohidGate @ hidGateError

      ResetGateGradient = self.resetgatederivative[i, :] * ResetGateError
      UpdateGateGradient = self.updategatederivative[i, :] * UpdateGateError
      hidGateGradient = self.hidgatederivative[i, :] * hidGateError

      weightsinToResetGate?? += np.outer(self.input[i, :].T, ResetGateGradient)
      weightsinToUpdateGate?? += np.outer(self.input[i, :].T, UpdateGateGradient)
      weightsinTohidGate?? += np.outer(self.input[i, :].T, hidGateGradient)
  
      weightshidToResetGate?? += np.outer(self.hid[i, :].T, ResetGateGradient)
      weightshidToUpdateGate?? += np.outer(self.hid[i, :].T, UpdateGateGradient)
      weightshidTohidGate?? += np.outer(self.hid[i, :].T, hidGateGradient)
  
      hidGatebias?? += ResetGateGradient
      ResetGatebias?? += UpdateGateGradient
      UpdateGatebias?? += hidGateGradient

      weightshidToout?? += np.outer(self.hid[i].T, outGradient)
      outbias?? += outGradient

    self.weightsinToResetGate += np.clip(weightsinToResetGate??, -1, 1) * self.learningrate
    self.weightsinToUpdateGate += np.clip(weightsinToUpdateGate??, -1, 1) * self.learningrate
    self.weightsinTohidGate += np.clip(weightsinTohidGate??, -1, 1) * self.learningrate

    self.weightshidToResetGate += np.clip(weightshidToResetGate??, -1, 1) * self.learningrate
    self.weightshidToUpdateGate += np.clip(weightshidToUpdateGate??, -1, 1) * self.learningrate
    self.weightshidTohidGate += np.clip(weightshidTohidGate??, -1, 1) * self.learningrate

    self.hidGatebias += np.clip(hidGatebias??, -1, 1) * self.learningrate
    self.ResetGatebias += np.clip(ResetGatebias??, -1, 1) * self.learningrate
    self.UpdateGatebias += np.clip(UpdateGatebias??, -1, 1) * self.learningrate

    self.weightshidToout += np.clip(weightshidToout??, -1, 1) * self.learningrate
    self.outbias += np.clip(outbias??, -1, 1) * self.learningrate

    return inError