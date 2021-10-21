from Activation import *
from NN import NN
import numpy as np

def SingleHeadSelfAttention(In):
  InToKey = NN(InSize=In.shape, OutSize=In.shape, Activation='Identity', LearningRate=0)
  InToQuery = NN(InSize=In.shape, OutSize=In.shape, Activation='Identity', LearningRate=0)
  InToValue = NN(InSize=In.shape, OutSize=In.shape, Activation='Identity', LearningRate=0)

  Key = InToKey.ForwardProp(In)
  Query = InToQuery.ForwardProp(In)
  Value = InToValue.ForwardProp(In)

  Out = np.dot(Query, Key.T) # MatMul
  Out = Out / In.shape[1] ** 0.5 # Scale
  
  Out[np.triu_indices(Out.shape[0], 1)] = float('-inf') # Mask (opt.)

  Out = ApplyActivation(Out, "StableSoftmax") # SoftMax
  Out = Out @ Value # MatMul

  return Out

def MultiHeadSelfAttention(In, NumOfHeads):
  InToKey = NN(InShape=In.shape, OutShape=(In.shape[0], In.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  InToQuery = NN(InShape=In.shape, OutShape=(In.shape[0], In.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  InToValue = NN(InShape=In.shape, OutShape=(In.shape[0], In.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  KeyQueryValueToOut = NN(InShape=(In.shape[0], In.shape[1] * NumOfHeads), OutShape=In.shape, Activation='Identity', LearningRate=0)

  Key = InToKey.ForwardProp(In)
  Query = InToQuery.ForwardProp(In)
  Value = InToValue.ForwardProp(In)

  Out = np.dot(Query, Key.T) # MatMul
  Out = Out / In.shape[1] ** 0.5 # Scale
  
  Out[np.triu_indices(Out.shape[0], 1)] = float('-inf') # Mask (opt.)

  Out = ApplyActivation(Out, "StableSoftmax") # SoftMax
  Out = Out @ Value # MatMul

  Out = KeyQueryValueToOut.ForwardProp(Out)
  return Out

Input = np.array([[1, 0, 0], [0, 0, 1]])
print(MultiHeadSelfAttention(Input, 8))