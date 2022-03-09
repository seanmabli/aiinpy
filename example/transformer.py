import src as ai
import numpy as np

def basicselfattention(input):
  weights = np.dot(input, input.T)
  weights = ai.stablesoftmax().forward(weights)
  out = weights @ input
  return out

def singleheadselfattention(input):
  inToKey = ai.nn(inshape=input.shape, outshape=input.shape, activation=ai.identity, learningrate=0)
  inToQuery = ai.nn(inshape=input.shape, outshape=input.shape, activation=ai.identity, learningrate=0)
  inToValue = ai.nn(inshape=input.shape, outshape=input.shape, activation=ai.identity, learningrate=0)

  Key = inToKey.forward(input)
  Query = inToQuery.forward(input)
  Value = inToValue.forward(input)

  out = np.dot(Query, Key.T) # MatMul
  out = out / input.shape[1] ** 0.5 # Scale
  
  out[np.triu_indices(out.shape[0], 1)] = float('-inf') # Mask (opt.)

  out = ai.stablesoftmax().forward(out) # SoftMax
  out = out @ Value # MatMul

  return out

def multiheadselfattention(input, NumOfHeads):
  inToKey = ai.nn(inshape=input.shape, outshape=(input.shape[0], input.shape[1] * NumOfHeads), activation=ai.identity, learningrate=0)
  inToQuery = ai.nn(inshape=input.shape, outshape=(input.shape[0], input.shape[1] * NumOfHeads), activation=ai.identity, learningrate=0)
  inToValue = ai.nn(inshape=input.shape, outshape=(input.shape[0], input.shape[1] * NumOfHeads), activation=ai.identity, learningrate=0)
  KeyQueryValueToout = ai.nn(inshape=(input.shape[0], input.shape[1] * NumOfHeads), outshape=input.shape, activation=ai.identity, learningrate=0)

  Key = inToKey.forward(input)
  Query = inToQuery.forward(input)
  Value = inToValue.forward(input)

  out = np.dot(Query, Key.T) # MatMul
  out = out / input.shape[1] ** 0.5 # Scale
  
  out[np.triu_indices(out.shape[0], 1)] = float('-inf') # Mask (opt.)

  out = ai.stablesoftmax().forward(out) # SoftMax
  out = out @ Value # MatMul

  out = KeyQueryValueToout.forward(out)
  return out

input = np.array([[1, 0, 0], [0, 0, 1]])
print(basicselfattention(input))
print(singleheadselfattention(input))
print(multiheadselfattention(input, 8))