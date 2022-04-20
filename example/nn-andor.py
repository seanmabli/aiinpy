import src as ai
import numpy as np
# import wandb

# wandb.init(project="nn-andor")

# Create Dataset
inTrainData = np.random.choice(([0, 1]), (2, 100))
outTrainData = np.zeros((2, 100))
for i in range(100):
  outTrainData[:, i] = [1, 0] if sum(inTrainData[:, i]) == 1 else [0, 1]

# NN model
model = ai.model(inshape=2, outshape=2, model=[
  ai.nn(outshape=16, activation=ai.relu(), learningrate=0.01),
  ai.nn(outshape=16, activation=ai.relu(), learningrate=0.01),
  ai.nn(outshape=2, activation=ai.stablesoftmax(), learningrate=0.01)
])

model.train((inTrainData, outTrainData), 12000)
 #wandb.log({"test accuracy": model.test((inTrainData, outTrainData))})
print(model.test((inTrainData, outTrainData)))
print(np.around(model.use(inTrainData), 2))

model.exportcache()