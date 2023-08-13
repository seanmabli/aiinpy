import src as ai
import numpy as np

# Create Dataset
inTrainData = np.random.choice(([0, 1]), (2, 100))
outTrainData = np.zeros((2, 100))
for i in range(100):
  outTrainData[:, i] = [1, 0] if sum(inTrainData[:, i]) == 1 else [0, 1]

# NN model
model = ai.model(inshape=2, outshape=2, layers=[
  ai.nn(outshape=16, activation=ai.sigmoid(), learningrate=0.01),
  ai.nn(outshape=16, activation=ai.sigmoid(), learningrate=0.01),
  ai.nn(outshape=2, activation=ai.sigmoid(), learningrate=0.01)
])

model.train((inTrainData, outTrainData), 12000)
model.test((inTrainData, outTrainData))