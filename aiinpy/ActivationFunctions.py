import numpy as np

class Sigmoid:
  def __init__(self, Input):
    return 1 / (1 + np.exp(-Input))
  def Derivative(self, Input):
    return Input * (1 - Input)

class Tanh:
  def __init__(Input):
    return np.tanh(Input)
  def Derivative(Input):
    return 1 - np.square(Input)

class ReLU:
  def __init__(Input):
    Output = np.maximum(0, Input)
    return Output
  def Derivative(Input):
    Equation = np.vectorize(EquationForDerivativeOfReLU, otypes=[float])
    return Equation(Input)
  def EquationForDerivative(Input):
    return 0 if (Input <= 0) else 1

class LeakyReLU:
  def __init__(Input):
    return np.maximum(0.01 * Input, Input)
  def Derivative(Input):
    Equation = np.vectorize(EquationForDerivativeOfLeakyReLU, otypes=[float])
    return Equation(Input)
  def EquationForDerivative(Input):
    return 0.01 if (Input < 0) else 1

class StableSoftMax:
  def __init__(Input):
    return np.exp(Input - np.max(Input)) / np.sum(np.exp(Input - np.max(Input)))
  def Derivative(Input):
    Equation = np.vectorize(EquationForDerivativeOfStableSoftMax, otypes=[float])
    return Equation(Input, np.sum(np.exp(Input)))
  def EquationForDerivative(Input, SumExpOfInput):
    return (np.exp(Input) * (SumExpOfInput - np.exp(Input)))/(SumExpOfInput) ** 2