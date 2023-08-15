from emnist import extract_training_samples, extract_test_samples
import src as ai

# Create Dataset
inTrainData, outTrainData = extract_training_samples('bymerge')
inTestData, outTestData = extract_test_samples('bymerge')
inTrainData, inTestData = (inTrainData[0 : 10000] / 255) - 0.5, (inTestData[0 : 1000] / 255) - 0.5

outTrainDataReal = ai.tensor.zeros((10000, 47))
for i in range(10000):
  outTrainDataReal[i, outTrainData[i]] = 1
outTestDataReal = ai.tensor.zeros((1000, 47))
for i in range(1000):
  outTestDataReal[i, outTestData[i]] = 1

# CNN model
model = ai.model((28, 28), 47, [
  ai.conv(numoffilters=4, activation=ai.relu(), filtershape=(3, 3), learningrate=0.01, padding=True),
  ai.pool(stride=(2, 2), filtershape=(2, 2), operation='max'),
  ai.nn(outshape=47, activation=ai.stablesoftmax(), learningrate=0.1, weightsinit=(0, 0))
], "cnn-emnist-bymerge")

model.train((inTrainData, outTrainDataReal), 5000)
print(model.test((inTestData, outTestDataReal)))