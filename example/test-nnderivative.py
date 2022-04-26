import numpy as np
import src as newai
import old as oldai

inTrainData = np.random.choice(([0, 1]), (2, 100))
outTrainData = np.zeros((2, 100))
for i in range(100):
  outTrainData[:, i] = [1, 0] if sum(inTrainData[:, i]) == 1 else [0, 1]

yl = 0
xl = 0

for _ in range(1000):
  activation = np.random.choice([newai.relu(), newai.leakyrelu(), newai.mish(), newai.silu(), newai.selu(), newai.softplus(), newai.stablesoftmax(), newai.binarystep(), newai.gaussian(), newai.identity()], 3)

  newmodel = newai.model(inshape=2, outshape=2, model=[
    newai.nn(outshape=16, activation=activation[0], learningrate=0.01),
    newai.nn(outshape=16, activation=activation[1], learningrate=0.01),
    newai.nn(outshape=2, activation=activation[2], learningrate=0.01)
  ])

  oldmodel = oldai.model(inshape=2, outshape=2, model=[
    oldai.nn(outshape=16, activation=activation[0], learningrate=0.01),
    oldai.nn(outshape=16, activation=activation[1], learningrate=0.01),
    oldai.nn(outshape=2, activation=activation[2], learningrate=0.01)
  ])

  y = np.sum(newmodel.train((inTrainData, outTrainData), 1000))
  x = 0
  for i in oldmodel.train((inTrainData, outTrainData), 1000):
    x += np.sum(i)

  yl += y
  xl += x
  print(newai.sigmoid().forward(y), newai.sigmoid().forward(x))

print(yl, xl)