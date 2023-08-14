from .tensor import tensor

class binarystep:
  def __repr__(self):
    return 'binarystep()'

  def forward(self, input):
    equation = tensor.vectorize(self.equationForbinaryStep)
    return equation(input)

  def equationForbinaryStep(self, input):
    return 0 if (input < 0) else 1
    
  def backward(self, input):
    return 0
    # if input == 0 return undefined, this was removed because it would return an error 

class elu:
  def __init__(self, alpha):
    self.alpha = alpha
    self.forwardequationvectorized = tensor.vectorize(self.forwardequation)
    self.backwardequationvectorized = tensor.vectorize(self.backwardequation)

  def __repr__(self):
    return 'elu(' + str(self.alpha) + ')'

  def forward(self, input):
    return self.forwardequationvectorized(input)

  def forwardequation(self, input):
    return self.alpha * (tensor.exp(input) - 1) if input <= 0 else input

  def backward(self, input):
    return self.backwardequationvectorized(input)

  def backwardequation(self, input):
    if input < 0:
      return self.alpha * tensor.exp(input)
    elif input > 0:
      return 1
    elif input == 0 and self.alpha == 1: # NOTE: what happends if alpha != 1?
      return 1

class gaussian:
  def __repr__(self):
    return 'gaussian()'

  def forward(self, input):
    return tensor.exp(-(input ** 2))
    
  def backward(self, input):
    return -2 * input * tensor.exp(-(input ** 2))
  
class identity:
  def __repr__(self):
    return 'identity()'

  def forward(self, input):
    return input
    
  def backward(self, input):
    return 1
  
class leakyrelu:
  def __init__(self, alpha=0.01):
    self.alpha = alpha

  def __repr__(self):
    return 'leakyrelu(' + str(self.alpha) + ')'

  def forward(self, input):
    return tensor.maximum(self.alpha * input, input)

  def backward(self, input):
    equation = tensor.vectorize(self.equationforderivative)
    return equation(input)

  def equationforderivative(self, input):
    return self.alpha if input < 0 else 1

class dropout:
  def __init__(self, dropoutrate):
    self.dropoutrate = dropoutrate

  def __copy__(self):
    return type(self)(self.dropoutrate)

  def __repr__(self):
    return 'dropout(dropoutrate=' + str(self.dropoutrate) + ')'

  def modelinit(self, inshape):
    return inshape

  def forward(self, input):
    self.out = tensor.random_binomial(input.shape, self.dropoutrate)
    self.out = tensor.where(self.out == 0, 1, 0)
    return self.out * input

  def backward(self, outError):
    return self.out * outError  
    
  def changeDropoutRate(self, NewRate):
    self.dropoutrate = NewRate

class mean:
  def __init__(self, axis=0):
      self.axis = axis

  def __repr__(self):
    return 'mean()'

  def modelinit(self, inshape):
    self.inshape = inshape
    return inshape[:self.axis] + inshape[self.axis + 1:]
    
  def forward(self, input):
    return input.mean(axis=self.axis)

  def backward(self, input): # fix
    # return tensor(tensor.ones(self.inshape) / self.inshape[self.axis])
    # return tensor.array([input] * self.inshape[self.axis])
    return input # NOTE: fix

class mish:
  def __repr__(self):
    return 'mish()'

  def forward(self, input):
    return (input * ((2 * tensor.exp(input)) + tensor.exp(2 * input))) / ((2 * tensor.exp(input)) + tensor.exp(2 * input) + 2)

  def backward(self, input):
    return (tensor.exp(input) * ((4 * tensor.exp(2 * input)) + tensor.exp(3 * input) + (4 * (1 + input)) + (tensor.exp(input) * (6 + (4 * input))))) / tensor.square(2 + (2 * tensor.exp(input)) + tensor.exp(2 * input))
  
class prelu:
  def __init__(self, alpha):
    self.alpha = alpha
    self.forwardequationvectorized = tensor.vectorize(self.forwardequation)
    self.backwardequationvectorized = tensor.vectorize(self.backwardequation)

  def __repr__(self):
    return 'prelu(' + str(self.alpha) + ')'

  def forward(self, input):
    return self.forwardequationvectorized(input)

  def forwardequation(self, input):
    return input if input >= 0 else self.alpha * input

  def backward(self, input):
    return self.backwardequationvectorized(input)

  def backwardequation(self, input):
    return 1 if input >= 0 else self.alpha

class relu:
  def __repr__(self):
    return 'relu()'

  def forward(self, input):
    output = tensor.maximum(0, input)
    return output

  def backward(self, input):
    equation = tensor.vectorize(self.equationforderivative)
    return equation(input)
    
  def equationforderivative(self, input):
    return 0 if input <= 0 else 1

class selu:
  def __repr__(self):
    return 'selu()'

  def forward(self, input):
    equation = tensor.vectorize(self.equationforselu)
    return 1.0507 * equation(input)

  def equationforselu(self, input):
    return 1.67326 * (tensor.exp(input) - 1) if (input < 0) else input

  def backward(self, input):
    equation = tensor.vectorize(self.equationforderivative)
    return 1.0507 * equation(input)

  def equationforderivative(self, input):
    return 1.67326 * tensor.exp(input) if (input < 0) else 1


class sigmoid:
  def __repr__(self):
    return 'sigmoid()'

  def forward(self, input):
    return 1 / (1 + tensor.exp(-input))
    
  def backward(self, input):
    return tensor.exp(-input) / ((1 + tensor.exp(-input)) ** 2)
  
class silu:
  def __repr__(self):
    return 'silu()'

  def forward(self, input):
    return input / (1 + tensor.exp(-input))
    
  def backward(self, input):
    return (1 + tensor.exp(-input) + (input * tensor.exp(-input))) / tensor.square(1 + tensor.exp(-input))

class softmax:
  def __repr__(self):
    return 'softmax()'

  def forward(self, input):
    return tensor.exp(input) / tensor.sum(tensor.exp(input))

  def backward(self, input):
    forward = tensor.exp(input) / tensor.sum(tensor.exp(input))
    out = tensor.zeros((len(input), len(input)))
    for i in range(len(input)):
      for j in range(len(input)):
        out[i, j] = forward[i] * (1 if i == j else 0 - forward[j])
    return out

class softplus:
  def __repr__(self):
    return 'softplus()'

  def forward(self, input):
    return tensor.log(1 + tensor.exp(input))
    
  def backward(self, input):
    return 1 / (1 + tensor.exp(-input))

class stablesoftmax:
  def __repr__(self):
    return 'stablesoftmax()'

  def forward(self, input):
    a = tensor.exp(input - tensor.max(input))
    return a / tensor.sum(a)

  def newbackward(self, input): # use this
    forward = tensor.exp(input - tensor.max(input)) / tensor.sum(tensor.exp(input - tensor.max(input)))
    out = tensor.zeros((len(input), len(input)))
    for i in range(len(input)):
      for j in range(len(input)):
        out[i, j] = forward[i] * (1 if i == j else 0 - forward[j])
    return out

  def newbackwardtwo(self, input):
    input = input.flatten()
    s = tensor.exp(input - tensor.max(input)) / tensor.sum(tensor.exp(input - tensor.max(input)))
    print(s.shape)
    a = tensor.eye(s.shape[-1])
    temp1 = tensor.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=tensor.float32)
    temp2 = tensor.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=tensor.float32)
    temp1 = tensor.einsum('ij,jk->ijk',s,a)
    temp2 = tensor.einsum('ij,ik->ijk',s,s)
    return temp1-temp2

  def backward(self, input):
    return (tensor.exp(input) * (tensor.sum(tensor.exp(input)) - tensor.exp(input))) / tensor.sum(tensor.exp(input)) ** 2

class tanh:
  def __repr__(self):
    return 'tanh()'

  def forward(self, input):
    a = tensor.exp(2 * input)
    return (4 * a) / (a + 1)

  def backward(self, input):
    a = tensor.exp(2 * input)
    return (4 * a) / ((a + 1) ** 2)