import src as ai
import numpy as np

model = ai.rnn(inshape=1, outshape=1, type='ManyToMany', outactivation=ai.identity(), learningrate=0.01)

Data = np.genfromtxt(r"data/timeseries/airpassenger.csv", dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

traingens = 15000
testgens = len(TestData) - 5

for gen in range(traingens):
  Random = np.random.randint(0, len(TrainingData) - 5)

  input = TrainingData[Random : Random + 5]
  Out = model.forward(input)

  OutError = TrainingData[Random + 1 : Random + 6] - Out
  inError = model.backward(OutError)

error = 0
for gen in range(testgens):
  input = TestData[gen : gen + 5]
  Out = model.forward(input)

  OutError = TestData[gen + 1 : gen + 6] - Out
  error += np.sum(OutError)
  inError = model.backward(OutError)

print(error / testgens)