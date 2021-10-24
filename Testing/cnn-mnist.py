import numpy as np
from emnist import extract_training_samples, extract_test_samples
from testsrc.nn import nn
from testsrc.conv import conv
from testsrc.pool import pool
from testsrc.model import model

# Create Dataset
InTrainData, OutTrainData = extract_training_samples('digits')
InTestData, OutTestData = extract_test_samples('digits')
InTrainData, InTestData = (InTrainData[0 : 5000] / 255) - 0.5, (InTestData[0 : 1000] / 255) - 0.5

OutTrainDataReal = np.zeros((5000, 10))
for i in range(5000):
  OutTrainDataReal[i, OutTrainData[i]] = 1
OutTestDataReal = np.zeros((1000, 10))
for i in range(1000):
  OutTestDataReal[i, OutTestData[i]] = 1

# CNN Model
model = model((28, 28), 10, [
  conv((4, 3, 3), 0.01, 'ReLU', True),
  pool((2, 2), (2, 2), 'Max'),
  nn((4, 14, 14), 10, 'StableSoftmax', 0.1, (0, 0))
])

model.train(InTrainData, OutTrainDataReal)
print(model.test(InTestData, OutTestDataReal))