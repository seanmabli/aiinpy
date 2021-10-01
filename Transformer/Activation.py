import numpy as np

'''
Need to work on:
- LeakyReLU
'''

def ApplyActivation(Input, Activation):
  return eval(str(Activation) + "()." + str(Activation) + "(Input)")

def ActivationDerivative(Input, Activation):
  return eval(str(Activation) + "().Derivative(Input)")

class Sigmoid:
  def Sigmoid(self, Input):
    return 1 / (1 + np.exp(-Input))
  def Derivative(self, Input):
    return Input * (1 - Input)

class Tanh:
  def Tanh(self, Input):
    return np.tanh(Input)
  def Derivative(self, Input):
    return 1 - np.square(Input)

class ReLU:
  def ReLU(self, Input):
    Output = np.maximum(0, Input)
    return Output
  def Derivative(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input)
  def EquationForDerivative(self, Input):
    return 0 if (Input <= 0) else 1

class LeakyReLU:
  def LeakyReLU(self, Input):
    return np.maximum(0.01 * Input, Input)
  def Derivative(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input)
  def EquationForDerivative(self, Input):
    if self.Slope is None:
      return 0.01 if (Input < 0) else 1
    else:
      return self.Slope if (Input < 0) else 1

class Identity:
  def Identity(self, Input):
    return Input
  def Derivative(self, Input):
    return 1
  
class BinaryStep:
  def BinaryStep(self, Input):
    Equation = np.vectorize(self.EquationForBinaryStep, otypes=[float])
    return Equation(Input)
  def EquationForBinaryStep(self, Input):
    return 0 if (Input < 0) else 1
  def Derivative(self, Input):
    return 1
    
'''
class GELU:
  def GELU(self, Input):
    return (0.5 * Input) * (1 + )
  def Derivative(self, Input):

class ELU:
  def ELU(self, Input):

  def Derivative(self, Input):
'''

class SELU:
  def SELU(self, Input):
    Equation = np.vectorize(self.EquationForSELU, otypes=[float])
    return 1.0507 * Equation(Input)
  def EquationForSELU(self, Input):
    return 1.67326 * (np.exp(Input) - 1) if (Input < 0) else Input
  def Derivative(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return 1.0507 * Equation(Input)
  def EquationForDerivative(self, Input):
    return 1.67326 * np.exp(Input) if (Input < 0) else 1

class SiLU:
  def SiLU(self, Input):
    return Input / (1 + np.exp(-Input))
  def Derivative(self, Input):
    return (1 + np.exp(-Input) + (Input * np.exp(-Input))) / np.square(1 + np.exp(-Input))
  
class Mish:
  def Mish(self, Input):
    return Input * Tanh().Tanh(np.log(1 + np.exp(Input)))
  def Derivative(self, Input):
    return (np.exp(Input) * ((4 * np.exp(2 * Input)) + np.exp(3 * Input) + (4 * (1 + Input)) + (np.exp(Input) * (6 + (4 * Input))))) / np.square(2 + (2 * np.exp(Input)) + np.exp(2 * Input))

class Gaussian:
  def Gaussian(self, Input):
    return np.exp(-np.square(Input))
  def Derivative(self, Input):
    return -2 * Input * np.exp(-np.square(Input))

class Softplus:
  def Softplus(self, Input):
    return np.log(1 + np.exp(Input))
  def Derivative(self, Input):
    return 1 / (1 + np.exp(-Input))

class Softmax:
  def Softmax(self, Input):
    return (Input - np.max(Input)) / np.sum(Input - np.max(Input)) # Check
  def Derivative(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input, np.sum(np.exp(Input)))
  def EquationForDerivative(self, Input, SumExpOfInput):
    return (np.exp(Input) * (SumExpOfInput - np.exp(Input)))/(SumExpOfInput) ** 2

class StableSoftmax:
  def StableSoftmax(self, Input):
    return np.exp(Input - np.max(Input)) / np.sum(np.exp(Input - np.max(Input)))
  def Derivative(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input, np.sum(np.exp(Input)))
  def EquationForDerivative(self, Input, SumExpOfInput):
    return (np.exp(Input) * (SumExpOfInput - np.exp(Input)))/(SumExpOfInput) ** 2