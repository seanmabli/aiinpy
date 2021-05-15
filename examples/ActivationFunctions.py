import numpy as np

def Sigmoid(Input):
  return 1 / (1 + np.exp(-Input))
def DerivativeOfSigmoid(Input):
  return Input * (1 - Input)

def StableSoftMax(Input):
  return np.exp(Input - np.max(Input)) / np.sum(np.exp(Input - np.max(Input)))
def DerivativeOfStableSoftMax(Input):
  Equation = np.vectorize(EquationForDerivativeOfStableSoftMax, otypes=[float])
  return Equation(Input, np.sum(np.exp(Input)))
def EquationForDerivativeOfStableSoftMax(Input, SumExpOfInput):
  return (np.exp(Input) * (SumExpOfInput - np.exp(Input)))/(SumExpOfInput) ** 2

def ReLU(Input):
  Output = np.maximum(0, Input)
  return Output
def DerivativeOfReLU(Input):
  Equation = np.vectorize(EquationForDerivativeOfReLU, otypes=[float])
  return Equation(Input)
def EquationForDerivativeOfReLU(Input):
  return 0 if (Input <= 0) else 1

def Tanh(Input):
  return np.tanh(Input)
def DerivativeOfTanh(Input):
  return 1 - np.square(Input)

def LeakyReLU(Input):
  return np.maximum(0.01 * Input, Input)
def DerivativeOfLeakyReLU(Input):
  Equation = np.vectorize(EquationForDerivativeOfLeakyReLU, otypes=[float])
  return Equation(Input)
def EquationForDerivativeOfLeakyReLU(Input):
  return 0.01 if (Input < 0) else 1