import numpy as np

def conv(input, filter):
  inshape = input.shape
  filtershape = filter.shape

  if len(inshape) == 2:
    inshape = tuple([filtershape[0]]) + inshape
  if(input.ndim == 2):
    input = np.stack(([input] * filtershape[0]))

  outshape = tuple([filtershape[0], inshape[1] - filtershape[1] + 1, inshape[2] - filtershape[2] + 1])
  out = np.zeros(outshape)

  for i in range(0, outshape[1]):
    for j in range(0, outshape[2]):
      out[:, i, j] = np.sum(np.multiply(input[:, i : i + filtershape[1], j : j + filtershape[2]], filter), axis=(1, 2))

  return out

def convmatrix(input, filter):
  inshape = input.shape
  filtershape = filter.shape

  outshape = tuple([filtershape[0], inshape[0] - filtershape[1] + 1, inshape[1] - filtershape[2] + 1])

  filtermatrix = np.zeros((np.prod(inshape), np.prod(outshape)))

  for f in range(outshape[0]):
    for i in range(outshape[1]):
      for j in range(outshape[2]):
        x = np.zeros(inshape)
        x[i : i + filtershape[1], j : j + filtershape[2]] = filter[f]
        filtermatrix[:, f * outshape[1] * outshape[2] + i * outshape[2] + j] = x.flatten()

  return input.flatten() @ filtermatrix

input = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
filter = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

convout = conv(input, filter).flatten()
convmatrixout = convmatrix(input, filter)

print(convout, '\n', convmatrixout)