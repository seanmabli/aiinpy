import numpy as np
from PIL import Image, ImageOps
import testsrc as ai
import wandb

wandb.init(project="cnn-rocketdataset")
config = wandb.config
config.numoffilters = 16
config.resizeratio = 6
config.conv1size = 3
config.conv2size = 3
config.nn1size = 256
config.nn2size = 64
config.convlr = 0.01
config.nnlr = 0.1

newshape = (int(960 / config.resizeratio), int(540 / config.resizeratio)) # orignal shape = (960, 540)
electronimages = np.zeros((23, newshape[1], newshape[0]))
falcon9images = np.zeros((72, newshape[1], newshape[0]))
soyuzimages = np.zeros((59, newshape[1], newshape[0]))
spaceshuttleimages = np.zeros((70, newshape[1], newshape[0]))

for i in range(23):
  electronimages[i] = np.array(ImageOps.grayscale(Image.open('C:\\Users\\smdro\\aiinpy\\testing\\data\\rocketdataset\\electron\\' + str(i) + '.png').resize(newshape)))
for i in range(72):
  falcon9images[i] = np.array(ImageOps.grayscale(Image.open('C:\\Users\\smdro\\aiinpy\\testing\\data\\rocketdataset\\falcon9\\' + str(i) + '.png').resize(newshape)))
for i in range(59):
  soyuzimages[i] = np.array(ImageOps.grayscale(Image.open('C:\\Users\\smdro\\aiinpy\\testing\\data\\rocketdataset\\soyuz\\' + str(i) + '.png').resize(newshape)))
for i in range(70):
  spaceshuttleimages[i] = np.array(ImageOps.grayscale(Image.open('C:\\Users\\smdro\\aiinpy\\testing\\data\\rocketdataset\\spaceshuttle\\' + str(i) + '.png').resize(newshape)))

imagein = np.concatenate((electronimages[:20], falcon9images[:70], soyuzimages[:55], spaceshuttleimages[:65]), axis=0)
imageout = np.concatenate((np.stack([1, 0, 0, 0] for _ in range(20)), np.stack([0, 1, 0, 0] for _ in range(70)), np.stack([0, 0, 1, 0] for _ in range(55)), np.stack([0, 0, 0, 1] for _ in range(65))))

# CNN model
model = ai.model((newshape[1], newshape[0]), 4, [
  ai.conv(filtershape=(config.numoffilters, config.conv1size, config.conv1size), learningrate=config.convlr, activation=ai.relu()),
  ai.conv(filtershape=(config.numoffilters, config.conv2size, config.conv2size), learningrate=config.convlr, activation=ai.relu()),
  ai.nn(outshape=config.nn1size, activation=ai.relu(), learningrate=config.nnlr, weightsinit=(0, 0)),
  ai.nn(outshape=config.nn2size, activation=ai.relu(), learningrate=config.nnlr, weightsinit=(0, 0)),
  ai.nn(outshape=4, activation=ai.stablesoftmax(), learningrate=config.nnlr, weightsinit=(0, 0))
])

error = model.train((imagein, imageout), 250)
for i in error:
  wandb.log({'error': i})
# print(model.test((imagein, imageout))) # wandb.log({'accuracy': model.test((inTestData, outTestDataReal))})