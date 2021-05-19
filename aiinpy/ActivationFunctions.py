import numpy as np

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

class StableSoftMax:
  def StableSoftMax(self, Input):
    return np.exp(Input - np.max(Input)) / np.sum(np.exp(Input - np.max(Input)))
  def Derivative(self, Input):
    Equation = np.vectorize(self.EquationForDerivative, otypes=[float])
    return Equation(Input, np.sum(np.exp(Input)))
  def EquationForDerivative(self, Input, SumExpOfInput):
    return (np.exp(Input) * (SumExpOfInput - np.exp(Input)))/(SumExpOfInput) ** 2