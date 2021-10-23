import numpy as np
from emnist import extract_training_samples, extract_test_samples
from testsrc.nn import nn
from testsrc.conv import conv
from testsrc.pool import pool
from testsrc.model import model

# Create Dataset
InTrainData, OutTrainData = extract_training_samples('digits')
InTestData, OutTestData = extract_test_samples('digits')[0 : 1000]

# CNN Model
model = model((4, 3, 3), 10, [
  conv((4, 3, 3), 0.01, 'ReLU', True),
  pool((2, 2), (2, 2), 'Max'),
  nn((4, 14, 14), 10, 'StableSoftmax', 0.1, (0, 0))
])

model.train(InTrainData, OutTrainData, 5000)
testcorrect = model.test(InTestData, OutTestData)
print(testcorrect)