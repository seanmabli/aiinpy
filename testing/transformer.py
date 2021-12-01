import testsrc as ai

def SingleHeadSelfattention(in):
  inToKey = NN(inshape=in.shape, outshape=in.shape, activation='Identity', learningrate=0)
  inToQuery = NN(inshape=in.shape, outshape=in.shape, activation='Identity', learningrate=0)
  inToValue = NN(inshape=in.shape, outshape=in.shape, activation='Identity', learningrate=0)

  Key = inToKey.forward(in)
  Query = inToQuery.forward(in)
  Value = inToValue.forward(in)

  out = np.dot(Query, Key.T) # MatMul
  out = out / in.shape[1] ** 0.5 # Scale
  
  out[np.triu_indices(out.shape[0], 1)] = float('-inf') # Mask (opt.)

  out = applyactivation(out, "StableSoftmax") # SoftMax
  out = out @ Value # MatMul

  return out

def MultiHeadSelfattention(in, NumOfHeads):
  inToKey = NN(inshape=in.shape, outshape=(in.shape[0], in.shape[1] * NumOfHeads), activation='Identity', learningrate=0)
  inToQuery = NN(inshape=in.shape, outshape=(in.shape[0], in.shape[1] * NumOfHeads), activation='Identity', learningrate=0)
  inToValue = NN(inshape=in.shape, outshape=(in.shape[0], in.shape[1] * NumOfHeads), activation='Identity', learningrate=0)
  KeyQueryValueToout = NN(inshape=(in.shape[0], in.shape[1] * NumOfHeads), outshape=in.shape, activation='Identity', learningrate=0)

  Key = inToKey.forward(in)
  Query = inToQuery.forward(in)
  Value = inToValue.forward(in)

  out = np.dot(Query, Key.T) # MatMul
  out = out / in.shape[1] ** 0.5 # Scale
  
  out[np.triu_indices(out.shape[0], 1)] = float('-inf') # Mask (opt.)

  out = applyactivation(out, "StableSoftmax") # SoftMax
  out = out @ Value # MatMul

  out = KeyQueryValueToout.forward(out)
  return out

input = np.array([[1, 0, 0], [0, 0, 1]])
print(MultiHeadSelfattention(input, 8))