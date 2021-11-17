import numpy as np

'''
- Add GeLU
- Add ELU
- Add PReLU
'''

class sigmoid:
  def forward(Input):
    return 1 / (1 + np.exp(-Input))
  def backward(Input):
    return Input * (1 - Input)

class tanh:
  def forward(Input):
    return np.tanh(Input)
  def backward(Input):
    return 1 - np.square(Input)

class relu:
  def forward(Input):
    Output = np.maximum(0, Input)
    return Output
  def backward(Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input)
  def EquationForDerivative(Input):
    return 0 if (Input <= 0) else 1

class leakyrelu:
  def __init__(self, alpha=0.01):
    self.alpha = alpha
  def forward(self, Input):
    return np.maximum(self.alpha * Input, Input)
  def backward(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input)
  def EquationForDerivative(self, Input):
    return self.alpha if Input < 0 else 1

class identity:
  def forward(self, Input):
    return Input
  def backward(self, Input):
    return 1
  
class binarystep:
  def forward(self, Input):
    Equation = np.vectorize(self.EquationForBinaryStep, otypes=[float])
    return Equation(Input)
  def EquationForBinaryStep(self, Input):
    return 0 if (Input < 0) else 1
  def backward(self, Input):
    return 1
    
class selu:
  def forward(self, Input):
    Equation = np.vectorize(self.EquationForSELU, otypes=[float])
    return 1.0507 * Equation(Input)
  def EquationForSELU(self, Input):
    return 1.67326 * (np.exp(Input) - 1) if (Input < 0) else Input
  def backward(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return 1.0507 * Equation(Input)
  def EquationForDerivative(self, Input):
    return 1.67326 * np.exp(Input) if (Input < 0) else 1

class silu:
  def forward(self, Input):
    return Input / (1 + np.exp(-Input))
  def backward(self, Input):
    return (1 + np.exp(-Input) + (Input * np.exp(-Input))) / np.square(1 + np.exp(-Input))
  
class mish:
  def forward(self, Input):
    return Input * Tanh().Tanh(np.log(1 + np.exp(Input)))
  def backward(self, Input):
    return (np.exp(Input) * ((4 * np.exp(2 * Input)) + np.exp(3 * Input) + (4 * (1 + Input)) + (np.exp(Input) * (6 + (4 * Input))))) / np.square(2 + (2 * np.exp(Input)) + np.exp(2 * Input))

class gaussian:
  def forward(self, Input):
    return np.exp(-np.square(Input))
  def backward(self, Input):
    return -2 * Input * np.exp(-np.square(Input))

class softplus:
  def forward(self, Input):
    return np.log(1 + np.exp(Input))
  def backward(self, Input):
    return 1 / (1 + np.exp(-Input))

class softmax:
  def forward(self, Input):
    return (Input - np.max(Input)) / np.sum(Input - np.max(Input)) # Check
  def backward(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input, np.sum(np.exp(Input)))
  def EquationForDerivative(self, Input, SumExpOfInput):
    return (np.exp(Input) * (SumExpOfInput - np.exp(Input)))/(SumExpOfInput) ** 2

class stablesoftmax:
  def forward(Input):
    return np.exp(Input - np.max(Input)) / np.sum(np.exp(Input - np.max(Input)))
  def backward(Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input, np.sum(np.exp(Input)))
  def EquationForDerivative(Input, SumExpOfInput):
    return (np.exp(Input) * (SumExpOfInput - np.exp(Input)))/(SumExpOfInput) ** 2