import aiinpy as ai
import numpy as np

def WordToBinary(Input):
  Dec = list(bytearray(Input, "utf8"))
  Output = [''] * len(Input)
  for i in range(len(Input)):
    Output[i] = bin(Dec[i]).replace("b", ("0"*(9-len(bin(Dec[i])))))
  return Output

class Transformer:
  def __init__(self, Heads, InputShape):
    self.ToKey = np.random.uniform(-np.sqrt(1 / InputShape), np.sqrt(1 / InputShape), (InputShape, InputShape))
    self.ToQuery = np.random.uniform(-np.sqrt(1 / InputShape), np.sqrt(1 / InputShape), (InputShape, InputShape))
    self.ToValue = np.random.uniform(-np.sqrt(1 / InputShape), np.sqrt(1 / InputShape), (InputShape, InputShape))
    self.UnifyHeads = np.random.uniform(-np.sqrt(1 / InputShape), np.sqrt(1 / InputShape), (InputShape * Heads, InputShape))
  
  def SelfAttention(self, Input):
    self.Weights = (Input @ np.transpose(Input)) / np.sqrt(Input.size)
    self.Weights = ai.StableSoftMax(self.Weights)
    self.Output = self.Weights @ Input
    return self.Output

Bob = Transformer(1, 8)

# Input To Binary
Input = "hello"
BinaryInput = np.zeros((len(WordToBinary(Input)), 8))
for i in range(len(WordToBinary(Input))):
  BinaryInput[i, :] = np.array(list(WordToBinary(Input)[i]))

Output = Bob.SelfAttention(BinaryInput)

'''
- Scaled self-attention might be scaled by the wrong dimention
- Multi-head self-attention
- Add key, query, value
'''