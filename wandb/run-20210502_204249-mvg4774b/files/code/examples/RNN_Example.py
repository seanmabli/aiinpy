import numpy as np
import aiinpy as ai
import wandb

wandb.init(project='rnn')

PositiveComments = open("C:\\Users\\smdro\\Downloads\\archive\\sentence_polarity\\rt-polarity.pos", "r")
NegativeComments = open("C:\\Users\\smdro\\Downloads\\archive\\sentence_polarity\\rt-polarity.neg", "r")
Comments = str(PositiveComments.read() + NegativeComments.read())
Comments = np.array(Comments.splitlines())[:100]

UniqueWords = list(set([w for Sentence in Comments for w in Sentence.split(' ')]))
Rnn = ai.RNN(len(UniqueWords), 2, "ManyToOne")

print(len(UniqueWords))

for Generation in range(100000):
  Random = np.random.randint(0, len(Comments))

  InputSentenceSplit = list(Comments[Random].split(' '))
  Input = np.zeros((len(InputSentenceSplit), len(UniqueWords)))
  for i in range(len(InputSentenceSplit)):
    Input[i, UniqueWords.index(InputSentenceSplit[i])] = 1

  Output = Rnn.ForwardProp(Input)

  RealOutput = np.array([1, 0]) if Random < 5331 else np.array([0, 1])
  NumberCorrect = int(np.argmax(Output) == (1 if Random < 5331 else 0))

  OutputError = RealOutput - Output

  Rnn.BackProp(OutputError)
  wandb.log({"Number Correct": NumberCorrect})