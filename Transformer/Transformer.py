from Activation import *
from NN import NN
import numpy as np

def SingleHeadSelfAttention(In):
  InToKey = NN(InSize=In.shape[1], OutSize=In.shape[1], Activation='Identity', LearningRate=0)
  InToQuery = NN(InSize=In.shape[1], OutSize=In.shape[1], Activation='Identity', LearningRate=0)
  InToValue = NN(InSize=In.shape[1], OutSize=In.shape[1], Activation='Identity', LearningRate=0)

  Key = np.zeros(In.shape)
  Query = np.zeros(In.shape)
  Value = np.zeros(In.shape)

  for i in range(In.shape[0]):
    Key[i, :] = InToKey.ForwardProp(In[i, :])
    Query[i, :] = InToQuery.ForwardProp(In[i, :])
    Value[i, :] = InToValue.ForwardProp(In[i, :])

  Out = np.dot(In, np.transpose(In))
  Out = Out / In.shape[1] ** 0.5
  
  Out[np.triu_indices(Out.shape[0], 1)] = float('-inf')
  
  Out = ApplyActivation(Out, "StableSoftmax")
  Out = Out @ Value

  return Out

def MultiHeadSelfAttention(Input, NumOfHeads):
  InToKey = NN(InSize=Input.shape[1], OutSize=(Input.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  InToQuery = NN(InSize=Input.shape[1], OutSize=(Input.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  InToValue = NN(InSize=Input.shape[1], OutSize=(Input.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  KeyQueryValueToOut = NN(InSize=(Input.shape[1] * NumOfHeads), OutSize=Input.shape[1], Activation='Identity', LearningRate=0)

  Key = np.zeros((Input.shape[0], Input.shape[1] * NumOfHeads))
  Query = np.zeros((Input.shape[0], Input.shape[1] * NumOfHeads))
  Value = np.zeros((Input.shape[0], Input.shape[1] * NumOfHeads))

  for i in range(Input.shape[0]):
    Key[i, :] = InToKey.ForwardProp(Input[i, :])
    Query[i, :] = InToQuery.ForwardProp(Input[i, :])
    Value[i, :] = InToValue.ForwardProp(Input[i, :])
  
  Out = np.dot(Query, np.transpose(Key))

  Out = ApplyActivation(Out, "StableSoftmax")
  Out = np.sum(Out)
  Out *= Value

  Out = KeyQueryValueToOut.ForwardProp(Out)
  return Out


Input = np.array([[1, 0, 0], [0, 0, 1]])
print(SingleHeadSelfAttention(Input))