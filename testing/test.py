import testsrc as ai
import numpy as np

conv = ai.convmatrix(inshape=(3, 3), filtershape=(1, 2, 2), filter=np.array([[[1, 2], [2, 1]]]), learningrate=0.01, activation=ai.identity())
input = np.array([[1, 2, 3], [6, 5, 3], [1, 4, 1]])
# conv.filtermatrix = np.array([[1, 2, 0, 2, 1, 0, 0, 0, 0], [0, 1, 2, 0, 2, 1, 0, 0, 0], [0, 0, 0, 1, 2, 0, 2, 1, 0], [0, 0, 0, 0, 1, 2, 0, 2, 1]]).T
print(conv.x)
out, outone = conv.forward(input)
print(out, outone)