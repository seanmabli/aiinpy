import aiinpy as ai
from alive_progress import alive_bar
import numpy as np
import wandb

model = ai.lstm(inshape=1, outshape=1, outactivation=ai.identity(), learningrate=0.01)

Data = np.genfromtxt(r'data\timeseries\airpassenger.csv', dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

NumOfTrainGen = 15000
NumOfTestGen = len(TestData) - 6

for _ in range(NumOfTrainGen):
  Random = np.random.randint(0, len(TrainingData) - 5)
  input = TrainingData[Random : Random + 5]  
  out = model.forward(input)
  
  outError = TrainingData[Random + 1 : Random + 6] - out
  inError = model.backward(outError)

Error = 0
for gen in range(NumOfTestGen):
  input = TestData[gen : gen + 5]
  out = model.forward(input)

  outError = TestData[gen + 1 : gen + 6] - out
  Error += abs(TestData[gen + 6] - out[4])
  inError = model.backward(outError)
  
print(Error / NumOfTestGen)