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
  Out /= In.shape[1] ** 0.5
  # Didn't include Mask (opt.)

  Out = ApplyActivation(Out, "StableSoftmax")
  Out = np.sum(Out)
  Out *= Value

  return Out

def MultiHeadSelfAttention(Input, NumOfHeads):
  InToKey = NN(InputSize=Input.shape[1], OutSize=(Input.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  InToQuery = NN(InputSize=Input.shape[1], OutSize=(Input.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  InToValue = NN(InputSize=Input.shape[1], OutSize=(Input.shape[1] * NumOfHeads), Activation='Identity', LearningRate=0)
  Key = np.zeros((Input.shape[0], Input.shape[1] * NumOfHeads))
  Query = np.zeros((Input.shape[0], Input.shape[1] * NumOfHeads))
  Value = np.zeros((Input.shape[0], Input.shape[1] * NumOfHeads))
  for i in range(Input.shape[0]):
    Key[i, :] = InToKey.ForwardProp(Input[i, :])
    Query[i, :] = InToQuery.ForwardProp(Input[i, :])
    Value[i, :] = InToValue.ForwardProp(Input[i, :])
  
  Weights = np.dot(Key, np.transpose(Query))

  Weights = ApplyActivation(Weights, "StableSoftmax")
  print(Weights)

Input = np.array([[1, 0, 0], [0, 0, 1]])
print(SingleHeadSelfAttention(Input))