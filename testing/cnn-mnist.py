import numpy as np
from emnist import extract_training_samples, extract_test_samples
import testsrc as ai
import wandb

wandb.init(project="cnn-mnist")

# Create Dataset
inTrainData, outTrainData = extract_training_samples('digits')
inTestData, outTestData = extract_test_samples('digits')
inTrainData, inTestData = (inTrainData[0 : 5000] / 255) - 0.5, (inTestData[0 : 1000] / 255) - 0.5

outTrainDataReal = np.zeros((5000, 10))
for i in range(5000):
  outTrainDataReal[i, outTrainData[i]] = 1
outTestDataReal = np.zeros((1000, 10))
for i in range(1000):
  outTestDataReal[i, outTestData[i]] = 1

# CNN Model
model = ai.model((28, 28), 10, [
  ai.conv((28, 28), (4, 26, 26), (3, 3), 0.01, ai.relu(), False),
  ai.pool((4, 26, 26), (2, 2), (2, 2), 'Max'),
  ai.nn((4, 13, 13), 10, ai.stablesoftmax(), 0.1, (0, 0))
])

model.train(inTrainData, outTrainDataReal, 5000)
wandb.log({'accuracy': model.test(inTestData, outTestDataReal)})