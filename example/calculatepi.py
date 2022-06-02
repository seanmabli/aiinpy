import src as ai
import numpy as np
import mpmath as m
import matplotlib.pyplot as plt
# import wandb

# wandb.init(project="pi")

m.mp.dps = 10000
pi = list(map(int, list(str(4 * m.atan(1)).replace('.', ''))))

data = np.zeros((10000, 10))

for i, value in enumerate(pi):
  data[i, value] = 1

print(data[0, :])

model = ai.rnn(inshape=10, outshape=10, type='ManyToMany', outactivation=ai.stablesoftmax(), learningrate=0.01)

NumOfTrainGen = 15000
NumOfTestGen = 200

for Generation in range(NumOfTrainGen):
  Random = np.random.randint(0, 9995)

  input = data[Random : Random + 5, :]
  Out = model.forward(input)

  OutError = data[Random + 1 : Random + 6, :] - Out
  # wandb.log({"out error" : OutError})
  inError = model.backward(OutError)

error = 0
for Generation in range(NumOfTestGen):
  input = data[Generation : Generation + 5, :]
  Out = model.forward(input)

  OutError = data[Generation + 1 : Generation + 6, :] - Out
  error += np.sum(OutError)
  inError = model.backward(OutError)

print(error / NumOfTestGen)
# wandb.log({"test accuracy" : error / NumOfTestGen})