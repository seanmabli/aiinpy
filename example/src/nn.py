from .tensor import tensor
from .static_ops import identity

class nn:
  def __init__(self, outshape, activation, learningrate, weightsinit=(-1, 1), biasesinit=(0, 0), inshape=None):
    self.weightsinit, self.biasesinit = weightsinit, biasesinit
    self.activation, self.learningrate = activation, learningrate
    self.inshape = inshape
    if inshape is not None:
      self.weights = tensor.uniform(weightsinit[0], weightsinit[1], (tensor.prod(inshape), tensor.prod(outshape)))
      self.biases = tensor.uniform(biasesinit[0], biasesinit[1], tensor.prod(outshape))
    self.outshape = outshape
    
  def __copy__(self):
    return type(self)(self.outshape, self.activation, self.learningrate, self.weightsinit, self.biasesinit, self.inshape)

  def __repr__(self):
    return 'nn(inshape=' + str(self.inshape) + ', outshape=' + str(self.outshape) + ', activation=' + str(self.activation.__repr__()) + ', learningrate=' + str(self.learningrate) + ', weightsinit=' + str(self.weightsinit) + ', biasesinit=' + str(self.biasesinit) + ')'

  def modelinit(self, inshape):
    if type(inshape) == tuple and len(inshape) == 1:
      inshape = inshape[0]
    self.inshape = inshape

    self.weights = tensor.uniform(self.weightsinit[0], self.weightsinit[1], (tensor.prod(inshape), tensor.prod(self.outshape)))
    self.biases = tensor.uniform(self.biasesinit[0], self.biasesinit[1], tensor.prod(self.outshape))
    return self.outshape

  def forward(self, input):
    self.input = input.flatten()
    out = self.weights.T @ self.input + self.biases
    self.out = self.activation.forward(out)
    self.derivative = self.activation.backward(out) # now it applys the derivative to the output without the activation function, check if this is right
    return self.out.reshape(self.outshape)

  def backward(self, outerror):
    outerror = outerror.flatten()
    outgradient = self.derivative * outerror if tensor.ndim(self.derivative) == 1 else self.derivative @ outerror
    inputerror = self.weights @ outerror
    self.biases += outgradient * self.learningrate
    self.weights += tensor.outer(self.input.T, outgradient) * self.learningrate
    return inputerror.reshape(self.inshape)