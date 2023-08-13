from .tensor import tensor
from .static_ops import tanh, stablesoftmax

class rnn:
  def __init__(self, outshape, type, outactivation=stablesoftmax(), hidshape=64, learningrate=0.05, inshape=None):
    self.learningrate, self.type, self.outactivation = learningrate, type, outactivation
    self.inshape, self.hidshape, self.outshape = inshape, hidshape, outshape
    
    if inshape is not None:
      self.weightsinTohid = tensor.uniform(-0.005, 0.005, (tensor.prod(hidshape), tensor.prod(inshape)))

    self.weightshidTohid = tensor.uniform(-0.005, 0.005, (tensor.prod(hidshape), tensor.prod(hidshape)))
    self.hidbiases = tensor.zeros(hidshape)

    self.weightshidToout = tensor.uniform(-0.005, 0.005, (tensor.prod(outshape), tensor.prod(hidshape)))
    self.outbiases = tensor.zeros(outshape)

  def __copy__(self):
    return type(self)(self.outshape, self.type, self.outactivation, self.hidshape, self.learningrate, self.inshape)

  def __repr__(self):
    return 'relu(inshape=' + str(self.inshape) + ', hidshape=' + str(self.hidshape) + ', outshape=' + str(self.outshape) + ', type=' + str(self.type) + ', outactivation=' + str(self.outactivation) + ', learningrate=' + str(self.learningrate) + ')'

  def modelinit(self, inshape):
    self.inshape = inshape
    self.weightsinTohid = tensor.uniform(-0.005, 0.005, (tensor.prod(self.hidshape), tensor.prod(inshape)))
    return self.outshape

  def forward(self, input):
    self.input = input
    self.hid = tensor.zeros((len(self.input) + 1, self.hidshape))
    self.hidderivative = tensor.zeros((len(self.input) + 1, self.hidshape))
    
    if self.type == 'ManyToOne':
      for i in range(len(input)):
        self.hid[i + 1, :] = tanh().forward(self.weightsinTohid @ input[i].flatten() + self.weightshidTohid @ self.hid[i, :] + self.hidbiases)
        self.hidderivative[i + 1, :] = tanh().backward(self.weightsinTohid @ input[i].flatten() + self.weightshidTohid @ self.hid[i, :] + self.hidbiases)

      self.out = self.outactivation.forward(self.weightshidToout @ self.hid[len(input), :] + self.outbiases)
      self.outderivative = self.outactivation.backward(self.weightshidToout @ self.hid[len(input), :] + self.outbiases)
    
    elif self.type == 'ManyToMany':
      self.out = tensor.zeros((len(self.input), self.outshape))
      self.outderivative = tensor.zeros((len(self.input), self.outshape))

      for i in range(len(input)):
        self.hid[i + 1, :] = tanh().forward(self.weightsinTohid @ input[i].flatten() + self.weightshidTohid @ self.hid[i, :] + self.hidbiases)
        self.hidderivative[i + 1, :] = tanh().backward(self.weightsinTohid @ input[i].flatten() + self.weightshidTohid @ self.hid[i, :] + self.hidbiases)
        self.out[i, :] = self.outactivation.forward(self.weightshidToout @ self.hid[i + 1, :] + self.outbiases)
        self.outderivative[i, :] = self.outactivation.backward(self.weightshidToout @ self.hid[i + 1, :] + self.outbiases)

    return self.out

  def backward(self, outError):
    weightsinTohidΔ = tensor.zeros(self.weightsinTohid.shape)
    weightshidTohidΔ = tensor.zeros(self.weightshidTohid.shape)
    hidbiasesΔ = tensor.zeros(self.hidbiases.shape)

    if self.type == 'ManyToOne':
      outGradient = self.outderivative * outError if tensor.ndim(self.outderivative) == 1 else self.outderivative @ outError

      weightshidTooutΔ = tensor.outer(outGradient, self.hid[len(self.input)].T)
      outbiasesΔ = outGradient

      hidError = self.weightshidToout.T @ outError

      for i in reversed(range(len(self.input))):
        hidGradient = self.hidderivative[i + 1, :] * hidError

        hidbiasesΔ += hidGradient
        weightshidTohidΔ += tensor.outer(hidGradient, self.hid[i].T)
        weightsinTohidΔ += tensor.outer(hidGradient, self.input[i].T)

        hidError = self.weightshidTohid.T @ hidGradient

    elif self.type == 'ManyToMany':
      weightshidTooutΔ = tensor.zeros(self.weightshidToout.shape)
      outbiasesΔ = tensor.zeros(self.outbiases.shape)

      hidError = self.weightshidToout.T @ outError[len(self.input) - 1]

      for i in reversed(range(len(self.input))):
        hidGradient = self.hidderivative[i + 1] * hidError
        outGradient = self.outderivative[i] * outError[i]

        weightsinTohidΔ += tensor.outer(hidGradient, self.input[i].T)
        weightshidTohidΔ += tensor.outer(hidGradient, self.hid[i].T)
        hidbiasesΔ += hidGradient

        weightshidTooutΔ += tensor.outer(outGradient, self.hid[i].T)
        outbiasesΔ += outGradient

        hidError = self.weightshidTohid.T @ hidGradient + self.weightshidToout.T @ outError[i]

    self.weightsinTohid += self.learningrate * tensor.clip(weightsinTohidΔ, -1, 1)
    self.weightshidTohid += self.learningrate * tensor.clip(weightshidTohidΔ, -1, 1)
    self.hidbiases += self.learningrate * tensor.clip(hidbiasesΔ, -1, 1)

    self.weightshidToout += self.learningrate * tensor.clip(weightshidTooutΔ, -1, 1)
    self.outbiases += self.learningrate * tensor.clip(outbiasesΔ, -1, 1)