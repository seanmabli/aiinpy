import pickle
import numpy as np
import src as ai

data_batch_1 = pickle.load(open("data/cifar10/data_batch_1", 'rb'), encoding='bytes')
test_batch = pickle.load(open("data/cifar10/test_batch", 'rb'), encoding='bytes')

inTrainData, outTrainData = data_batch_1[b'data'].reshape(10000, 3, 32, 32), data_batch_1[b'labels'][0:10000]
inTestData, outTestData = test_batch[b'data'][:1000].reshape(1000, 3, 32, 32), test_batch[b'labels'][:1000]
inTrainData, inTestData = (inTrainData / 255) - 0.5, (inTestData / 255) - 0.5

outTrainDataReal = np.zeros((10000, 10))
for i in range(10000):
  outTrainDataReal[i, outTrainData[i]] = 1
outTestDataReal = np.zeros((1000, 10))
for i in range(1000):
  outTestDataReal[i, outTestData[i]] = 1

# CNN model
model = ai.model((3, 32, 32), 10, [
  ai.conv(inshape=(3, 32, 32), filtershape=(3, 3, 3), learningrate=0.01, activation=ai.relu()),
  ai.pool(stride=(2, 2), filtershape=(2, 2), opperation='Max'),
  ai.nn(outshape=10, activation=ai.stablesoftmax(), learningrate=0.1, weightsinit=(0, 0))
])

model.train((inTrainData, outTrainDataReal), 3000)
print(model.test((inTestData, outTestDataReal)))