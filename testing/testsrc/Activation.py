import numpy as np

'''
- add GeLU
- add ELU
- add PReLU
'''

class sigmoid:
  def forward(self, input):
    return 1 / (1 + np.exp(-input))
  def backward(self, input):
    return input * (1 - input)

class tanh:
  def forward(self, input):
    return np.tanh(input)
  def backward(self, input):
    return 1 - np.square(input)

class relu:
  def forward(self, input):
    output = np.maximum(0, input)
    return output
  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return equation(input)
  def equationforderivative(self, input):
    return 0 if (input <= 0) else 1

class leakyrelu:
  def __init__(self, alpha=0.01):
    self.alpha = alpha
  def forward(self, input):
    return np.maximum(self.alpha * input, input)
  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return equation(input)
  def equationforderivative(self, input):
    return self.alpha if input < 0 else 1

class identity:
  def forward(self, input):
    return input
  def backward(self, input):
    return 1
  
class binarystep:
  def forward(self, input):
    equation = np.vectorize(self.equationForbinaryStep, otypes=[float])
    return equation(input)
  def equationForbinaryStep(self, input):
    return 0 if (input < 0) else 1
  def backward(self, input):
    return 1
    
class selu:
  def forward(self, input):
    equation = np.vectorize(self.equationForSELU, otypes=[float])
    return 1.0507 * equation(input)
  def equationForSELU(self, input):
    return 1.67326 * (np.exp(input) - 1) if (input < 0) else input
  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return 1.0507 * equation(input)
  def equationforderivative(self, input):
    return 1.67326 * np.exp(input) if (input < 0) else 1

class silu:
  def forward(self, input):
    return input / (1 + np.exp(-input))
  def backward(self, input):
    return (1 + np.exp(-input) + (input * np.exp(-input))) / np.square(1 + np.exp(-input))
  
class mish:
  def forward(self, input):
    return input * Tanh().Tanh(np.log(1 + np.exp(input)))
  def backward(self, input):
    return (np.exp(input) * ((4 * np.exp(2 * input)) + np.exp(3 * input) + (4 * (1 + input)) + (np.exp(input) * (6 + (4 * input))))) / np.square(2 + (2 * np.exp(input)) + np.exp(2 * input))

class gaussian:
  def forward(self, input):
    return np.exp(-np.square(input))
  def backward(self, input):
    return -2 * input * np.exp(-np.square(input))

class softplus:
  def forward(self, input):
    return np.log(1 + np.exp(input))
  def backward(self, input):
    return 1 / (1 + np.exp(-input))

class softmax:
  def forward(self, input):
    return (input - np.max(input)) / np.sum(input - np.max(input)) # Check
  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return equation(input, np.sum(np.exp(input)))
  def equationforderivative(self, input, SumExpOfinput):
    return (np.exp(input) * (SumExpOfinput - np.exp(input))) / (SumExpOfinput) ** 2

class stablesoftmax:
  def forward(self, input):
    return np.exp(input - np.max(input)) / np.sum(np.exp(input - np.max(input)))
  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return equation(input, np.sum(np.exp(input)))
  def equationforderivative(self, input, SumExpOfinput):
    return (np.exp(input) * (SumExpOfinput - np.exp(input))) / (SumExpOfinput) ** 2