import numpy as np
import testsrc as ai

# Create Dataset
InTrainData = np.random.choice(([0, 1]), (2, 100))
OutTrainData = np.zeros((2, 100))
for i in range(100):
  OutTrainData[:, i] = [1, 0] if sum(InTrainData[:, i]) == 1 else [0, 1]

# NN Model
model = ai.model(2, 2, [
  ai.nn(2, 16, ai.relu(), 0.1),
  ai.nn(16, 16, ai.relu(), 0.1),
  ai.nn(16, 2, ai.sigmoid(), 0.1)
])

model.train(InTrainData, OutTrainData, 100)
print(model.test(InTrainData, OutTrainData))