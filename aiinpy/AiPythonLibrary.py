import numpy as np

# Equations
def Sigmoid(Input):
  return 1 / (1 + np.exp(-Input))
def DerivativeOfSigmoid(Input):
  return Input * (1 - Input)

def StableSoftMax(Input):
  return np.exp(Input - np.max(Input)) / np.sum(np.exp(Input - np.max(Input)))
def DerivativeOfStableSoftMax(Input):
  Output = np.zeros(Input.shape)
  for i in range(Output.size):
    Output[i] = (np.exp(Input[i]) * (sum(np.exp(Input)) - np.exp(Input[i])))/(sum(np.exp(Input))) ** 2
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

def WordToBinary(Input):
  Dec = list(bytearray(Input, "utf8"))
  Output = [''] * len(Input)
  for i in range(len(Input)):
    Output[i] = bin(Dec[i]).replace("b", ("0"*(9-len(bin(Dec[i])))))
  return Output

class CONV:
  def __init__(self, FilterShape, LearningRate):
    self.Filter = np.random.uniform(-0.25, 0.25, (FilterShape))
    self.LearningRate = LearningRate

  def ForwardProp(self, InputImage):
    self.InputImage = InputImage
    self.OutputWidth = len(InputImage[0]) - (len(self.Filter[0, 0]) - 1)
    self.OutputHeight = len(InputImage) - (len(self.Filter[0]) - 1)
    NumberOfFilters = len(self.Filter)
    OutputArray = np.zeros((NumberOfFilters, self.OutputHeight, self.OutputWidth))
    for FilterNumber in range(NumberOfFilters):
      for i in range(self.OutputHeight):
        for j in range(self.OutputWidth):
          OutputArray[FilterNumber, i, j] = np.sum(np.multiply(InputImage[i : i + 3, j : j + 3], self.Filter[FilterNumber, :, :]))
    return OutputArray
  
  def BackProp(self, ConvolutionError):
    NumberOfFilters = len(self.Filter)
    FilterGradients = np.zeros((NumberOfFilters, 3, 3))
    for FilterNumber in range(NumberOfFilters):
      for i in range(self.OutputHeight):
        for j in range(self.OutputWidth):
          FilterGradients[FilterNumber, :, :] += self.InputImage[i : (i + 3), j : (j + 3)] * ConvolutionError[FilterNumber, i, j]
    self.Filter += FilterGradients * self.LearningRate

class POOL: 
  def __init__(self, Stride):
    self.Stride = Stride
    
  def ForwardProp(self, InputArray):
    self.InputArray = InputArray
    OutputWidth = int(len(InputArray[0, 0]) / 2)
    OutputHeight = int(len(InputArray[0]) / 2)
    NumberOfFilters = len(InputArray)
    self.OutputArray = np.zeros((NumberOfFilters, OutputHeight, OutputWidth))
    for FilterNumber in range(NumberOfFilters):
      for i in range(OutputHeight):
        for j in range(OutputWidth):
          self.OutputArray[FilterNumber, i, j] = np.max(InputArray[FilterNumber, i * self.Stride : i * self.Stride + 2, j * self.Stride : j * self.Stride + 2])
    return self.OutputArray

  def BackProp(self, CurrentMaxPoolingLayerError):
    OutputWidth = len(self.InputArray[0, 0])
    OutputHeight = len(self.InputArray[0])
    NumberOfFilters = len(self.InputArray)
    PreviousConvolutionalLayerError = np.zeros(self.InputArray.shape)
    for FilterNumber in range(NumberOfFilters):
      for i in range(OutputHeight):
        for j in range(OutputWidth):
          if(self.OutputArray[FilterNumber, int(i / 2), int(j / 2)] == self.InputArray[FilterNumber, i, j]):
            PreviousConvolutionalLayerError[FilterNumber, i, j] = CurrentMaxPoolingLayerError[FilterNumber, int(i / 2), int(j / 2)] 
          else:
            PreviousConvolutionalLayerError[FilterNumber, i, j] = 0
    return PreviousConvolutionalLayerError

class NN:
  def __init__(self, CurrentLayerShape, FollowingLayerShape, Activation, LearningRate, WeightsInit=(-1, 1)):
    self.Weights = np.random.uniform(WeightsInit[0], WeightsInit[1], (CurrentLayerShape, FollowingLayerShape))
    self.Biases = np.zeros(FollowingLayerShape)
    self.Activation = Activation
    self.LearningRate = LearningRate
  
  def ForwardProp(self, InputLayer):
    self.InputLayer = InputLayer
    self.Output = np.transpose(self.Weights) @ InputLayer + self.Biases
  
    # Apply Activation Function
    if(self.Activation == "ReLU"):
      self.Output = ReLU(self.Output)
    if(self.Activation == "Sigmoid"):
      self.Output = Sigmoid(self.Output)
    if(self.Activation == "StableSoftMax"):
      self.Output = StableSoftMax(self.Output)
    return self.Output

  def BackProp(self, FollowingLayerError):
    # Calculate Gradients
    FollowingLayerGradient = np.zeros(self.Output.shape)
    if(self.Activation == "ReLU"):
      FollowingLayerGradient = np.multiply(DerivativeOfReLU(self.Output), FollowingLayerError) * self.LearningRate
    if(self.Activation == "Sigmoid"):
      FollowingLayerGradient = np.multiply(DerivativeOfSigmoid(self.Output), FollowingLayerError) * self.LearningRate
    if(self.Activation == "StableSoftMax"):
      FollowingLayerGradient = np.multiply(DerivativeOfStableSoftMax(self.Output), FollowingLayerError) * self.LearningRate
      
    # Calculate Current Layer Error
    CurrentLayerError = self.Weights @ FollowingLayerError
      
    # Apply Deltas To The Weights And Biases
    self.Biases += FollowingLayerGradient
    self.Weights += np.outer(np.transpose(self.InputLayer), FollowingLayerGradient)
    return CurrentLayerError