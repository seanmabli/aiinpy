import numpy as np
from PIL import Image, ImageOps
import testsrc as ai
import wandb
import time

start = time.time()
electronimages = np.zeros((23, 960, 540))
falcon9images = np.zeros((72, 960, 540))
soyuzimages = np.zeros((59, 960, 540))
spaceshuttleimages = np.zeros((70, 960, 540))

for i in range(23):
  electronimages[i] = np.array(ImageOps.grayscale(Image.open('testing\\data\\rocketdataset\\electron\\' + str(i) + '.png'))).T
for i in range(72):
  falcon9images[i] = np.array(ImageOps.grayscale(Image.open('testing\\data\\rocketdataset\\falcon9\\' + str(i) + '.png'))).T
for i in range(59):
  soyuzimages[i] = np.array(ImageOps.grayscale(Image.open('testing\\data\\rocketdataset\\soyuz\\' + str(i) + '.png'))).T
for i in range(70):
  spaceshuttleimages[i] = np.array(ImageOps.grayscale(Image.open('testing\\data\\rocketdataset\\spaceshuttle\\' + str(i) + '.png'))).T

print(time.time() - start)
# wandb.init(project="cnn-rocketdataset")

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
  ai.convmatrix(inshape=(28, 28), filtershape=(4, 3, 3), learningrate=0.01, activation=ai.relu()),
  ai.pool(stride=(2, 2), filtershape=(2, 2), opperation='Max'),
  ai.nn(outshape=10, activation=ai.stablesoftmax(), learningrate=0.1, weightsinit=(0, 0))
])

model.train((inTrainData, outTrainDataReal), 5)
# print(model.test((inTestData, outTestDataReal))) # wandb.log({'accuracy': model.test((inTestData, outTestDataReal))})