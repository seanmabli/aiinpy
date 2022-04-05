import aiinpy as ai
import numpy as np
from data.posnegcon.VictorZhou import TrainingData, TestData
from alive_progress import alive_bar

TrainingDataUniqueWords = list(set([w for Sentence in TrainingData.keys() for w in Sentence.split(' ')]))
model = ai.rnn(inshape=len(TrainingDataUniqueWords), outshape=2, type='ManyToOne', outactivation=ai.stablesoftmax(), learningrate=0.05)

NumOfTrainGen = 15000
NumOfTestGen = len(list(TestData.items()))

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):
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
    bar()

  NumberCorrect = 0
  for Generation in range(NumOfTestGen):
    items = list(TestData.items())

    inputSentenceSplit = list(items[Generation][0].split(' '))
    input = np.zeros((len(inputSentenceSplit), len(TrainingDataUniqueWords)))
    for i in range(len(inputSentenceSplit)):
      input[i, TrainingDataUniqueWords.index(inputSentenceSplit[i])] = 1

    output = model.forward(input)
    NumberCorrect += int(np.argmax(output) == (1 if items[Generation][1] == True else 0))
    bar()

print(NumberCorrect / NumOfTestGen)