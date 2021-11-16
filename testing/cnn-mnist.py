import numpy as np
from emnist import extract_training_samples, extract_test_samples
import testsrc as ai

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
model = ai.model((28, 28), 10, [
  ai.conv((28, 28), (4, 3, 3), 0.01, ai.relu(), True),
  ai.pool((2, 2), (2, 2), 'Max'),
  ai.nn((4, 14, 14), 10, ai.stablesoftmax(), 0.1, (0, 0))
])

model.train(InTrainData, OutTrainDataReal)
print(model.test(InTestData, OutTestDataReal))