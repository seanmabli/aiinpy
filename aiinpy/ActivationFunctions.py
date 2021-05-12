import numpy as np
'''
- DerivativeOfStableSoftMax
- DerivativeOfReLU
'''
def Sigmoid(Input):
  return 1 / (1 + np.exp(-Input))
def DerivativeOfSigmoid(Input):
  return Input * (1 - Input)

def StableSoftMax(Input):
  return np.exp(Input - np.max(Input)) / np.sum(np.exp(Input - np.max(Input)))
def DerivativeOfStableSoftMax(Input):
  Output = np.zeros(Input.shape)
  for i in range(Output.size):
    Output[i] = (np.exp(Input[i]) * (np.sum(np.exp(Input)) - np.exp(Input[i])))/(np.sum(np.exp(Input))) ** 2
  return Output

def ReLU(Input):
  Output = np.maximum(0, Input)
  return Output
def DerivativeOfReLU(Input):
  Output = np.zeros(Input.size)
  for i in range(Input.size):
    Output[i] = 0 if (Input[i] <= 0) else 1
  return Output

def Tanh(Input):
  return np.tanh(Input)
def DerivativeOfTanh(Input):
  return 1 - np.square(Input)