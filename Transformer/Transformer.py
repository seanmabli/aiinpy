from Activation import *
from NN import NN
import numpy as np

def WordToBinary(Input):
  Dec = list(bytearray(Input, "utf8"))
  Output = [''] * len(Input)
  for i in range(len(Input)):
    Output[i] = bin(Dec[i]).replace("b", ("0"*(9-len(bin(Dec[i])))))
  return Output

def SingleHeadSelfAttention(Input):
  Key = Input
  Query = Input
  Value = Input

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

  Key /= Input.shape[1] ** 0.25
  Query /= Input.shape[1] ** 0.25
  Value /= Input.shape[1] ** 0.25

  # Key = Key.reshape((NumOfHeads, 2, 3))
  # Query = Query.reshape((NumOfHeads, 2, 3))
  # Value = Value.reshape((NumOfHeads, 2, 3))
  # Weights = np.zeros((NumOfHeads, 2, 2))
  
  Weights = np.dot(Key, np.transpose(Query))

  Weights = ApplyActivation(Weights, "StableSoftmax")
  print(Weights)
  # return np.dot(Weights, Value)

'''
class Transformer:
  def __init__(self, Heads, InputShape):
    self.ToKey = np.random.uniform(-np.sqrt(1 / InputShape), np.sqrt(1 / InputShape), (InputShape, InputShape))
    self.ToQuery = np.random.uniform(-np.sqrt(1 / InputShape), np.sqrt(1 / InputShape), (InputShape, InputShape))
    self.ToValue = np.random.uniform(-np.sqrt(1 / InputShape), np.sqrt(1 / InputShape), (InputShape, InputShape))
    self.UnifyHeads = np.random.uniform(-np.sqrt(1 / InputShape), np.sqrt(1 / InputShape), (InputShape * Heads, InputShape))

# Input To Binary
Input = "hello"
BinaryInput = np.zeros((len(WordToBinary(Input)), 8))
for i in range(len(WordToBinary(Input))):
  BinaryInput[i, :] = np.array(list(WordToBinary(Input)[i]))

Output = Bob.SelfAttention(BinaryInput)
'''

'''
- Scaled self-attention
- Multi-head self-attention
- Add key, query, value
'''

Input = np.array([[1, 0, 0], [0, 0, 1]])
# x = np.array([1, 0, 0, 0, 0, 1])
# Heads = 2
# print(x.reshape((Heads,int(len(x)/Heads))))
print(SelfAttention(Input, 1))