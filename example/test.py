import aiinpy as ai
from emnist import extract_training_samples, extract_test_samples
import numpy as np

a = ai.conv(inshape=(28, 28), filtershape=(4, 3, 3), learningrate=0.01, activation=ai.identity())
b = ai.convmatrix(inshape=(28, 28), filtershape=(4, 3, 3), learningrate=0.01, activation=ai.identity())

b.filter = a.filter

intrain, outtrain = extract_training_samples('digits')

intrain = (intrain / 255) - 0.5

aout = a.forward(intrain[0])
bout = b.forward(intrain[0])

print(np.sum(abs(aout)))
print(np.sum(abs(bout)))