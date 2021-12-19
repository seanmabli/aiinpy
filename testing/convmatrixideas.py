import numpy as np
import testsrc as ai

def generateconvmatrix(input, weight):
  out = []
  flattenweight = np.array(weight[0])
  for row in weight[1:]:
    flattenweight = np.append(flattenweight, np.zeros(input.shape[0] - weight.shape[0]))
    flattenweight = np.append(flattenweight, row)
  
  i = 0
  while i + len(flattenweight) <= np.prod(input.shape):
    y = np.zeros((np.prod(input.shape)))
    y[i : i + len(flattenweight)] = flattenweight
    out.append(y)
    if (np.prod(input.shape) - (i + len(flattenweight))) % input.shape[0] == 0:
      i += weight.shape[0]
    else:
      i += 1
  return np.array(out)

input = np.zeros((10, 10))

conv = ai.conv(inshape=input.shape, filtershape=(9, 9), learningrate=0)
convout = conv.forward(input)

weight = conv.Filter.reshape(9, 9)
convmatrix = generateconvmatrix(input, weight)
convmatrixout = convmatrix.dot(input.flatten()).reshape(convout.shape)

print(convout == convmatrixout)