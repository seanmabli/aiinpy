import numpy as np

input = np.zeros((4, 4))
weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def flattenweight(input, weight):
  flattenweight = np.array(weight[0])
  for row in weight[1:]:
    flattenweight = np.append(flattenweight, np.zeros(input.shape[0] - weight.shape[0]))
    flattenweight = np.append(flattenweight, row)
  return flattenweight
    
out = np.zeros((4, np.prod(input.shape))) # replace 4 with variable later
fw = flattenweight(input, weight)

i = 0
for x in range(input.shape[0]):
  out[x, i : i + len(fw)] = fw
  if (np.prod(input.shape) - (i + len(fw))) % input.shape[0] == 0:
    i += weight.shape[0]
  else:
    i += 1

print(out)