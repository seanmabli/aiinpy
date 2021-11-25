import numpy as np
from alive_progress import alive_bar
from data.posnegcon.VictorZhou import TrainingData, TestData
import aiinpy as ai

TrainingDataUniqueWords = list(set([w for Sentence in TrainingData.keys() for w in Sentence.split(' ')]))
rnn_model = ai.rnn(len(TrainingDataUniqueWords), 2, Type='ManyToOne', OutActivation='StableSoftmax', learningrate=0.05)

NumOfTrainGen = 15000
NumOfTestGen = len(list(TestData.items()))

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):
    items = list(TrainingData.items())
    Random = np.random.randint(0, len(TrainingDataUniqueWords))

    InputSentenceSplit = list(items[Random][0].split(' '))
    Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
    for i in range(len(InputSentenceSplit)):
      Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1

    Output = rnn_model.forward(Input)

    RealOutput = np.zeros(Output.shape)
    RealOutput[(1 if items[Random][1] == True else 0)] = 1

    NumberCorrect = int(np.argmax(Output) == (1 if items[Random][1] == True else 0))

    OutputError = RealOutput - Output
    rnn_model.backward(OutputError)
    bar()

  NumberCorrect = 0
  for Generation in range(NumOfTestGen):
    items = list(TestData.items())

    InputSentenceSplit = list(items[Generation][0].split(' '))
    Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
    for i in range(len(InputSentenceSplit)):
      Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1

    Output = rnn_model.forward(Input)
    NumberCorrect += int(np.argmax(Output) == (1 if items[Generation][1] == True else 0))
    bar()

print(NumberCorrect / NumOfTestGen)