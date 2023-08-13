import src as ai
import numpy as np
from data.posnegcon.VictorZhou import TrainingData, TestData
import sys

TrainingDataUniqueWords = list(set([w for Sentence in TrainingData.keys() for w in Sentence.split(' ')]))
model = ai.rnn(inshape=len(TrainingDataUniqueWords), outshape=2, type='ManyToOne', learningrate=0.05)

NumOfTrainGen = 15000
NumOfTestGen = len(list(TestData.items()))

for gen in range(NumOfTrainGen):
  items = list(TrainingData.items())
  Random = np.random.randint(0, len(TrainingDataUniqueWords))

  inputSentenceSplit = list(items[Random][0].split(' '))
  input = np.zeros((len(inputSentenceSplit), len(TrainingDataUniqueWords)))
  for i in range(len(inputSentenceSplit)):
    input[i, TrainingDataUniqueWords.index(inputSentenceSplit[i])] = 1

  output = model.forward(input)

  Realoutput = np.zeros(output.shape)
  Realoutput[(1 if items[Random][1] == True else 0)] = 1

  NumberCorrect = int(np.argmax(output) == (1 if items[Random][1] == True else 0))

  outputError = Realoutput - output
  model.backward(outputError)

  sys.stdout.write('\r' + 'training: ' + str(gen + 1) + '/' + str(NumOfTrainGen))

print('')

NumberCorrect = 0
for gen in range(NumOfTestGen):
  items = list(TestData.items())

  inputSentenceSplit = list(items[gen][0].split(' '))
  input = np.zeros((len(inputSentenceSplit), len(TrainingDataUniqueWords)))
  for i in range(len(inputSentenceSplit)):
    input[i, TrainingDataUniqueWords.index(inputSentenceSplit[i])] = 1

  output = model.forward(input)
  NumberCorrect += int(np.argmax(output) == (1 if items[gen][1] == True else 0))

  sys.stdout.write('\r' + 'testing: ' + str(gen + 1) + '/' + str(NumOfTestGen))

print('\n' + str(NumberCorrect / NumOfTestGen))