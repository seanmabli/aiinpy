import numpy as np
from emnist import extract_training_samples, extract_test_samples
import src as ai

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

# CNN model
model = ai.model((28, 28), 10, [
  ai.conv(inshape=(28, 28), filtershape=(4, 3, 3), learningrate=0.01, activation=ai.relu()),
  ai.pool(stride=(2, 2), filtershape=(2, 2), opperation='Max'),
  ai.nn(outshape=10, activation=ai.stablesoftmax(), learningrate=0.1, weightsinit=(0, 0))
], wandbproject='cnn-mnist', usebestcache=True)

model.train((inTrainData, outTrainDataReal), 5000)
model.test((inTestData, outTestDataReal))