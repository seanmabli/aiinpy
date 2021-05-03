import numpy as np
import aiinpy as ai
import wandb

wandb.init(project='rnn')

PositiveComments = open("C:\\Users\\smdro\\Downloads\\archive\\sentence_polarity\\rt-polarity.pos", "r")
NegativeComments = open("C:\\Users\\smdro\\Downloads\\archive\\sentence_polarity\\rt-polarity.neg", "r")
Comments = str(PositiveComments.read() + NegativeComments.read())[:1000]
Comments = np.array(Comments.splitlines())

UniqueWords = list(set([w for Sentence in Comments for w in Sentence.split(' ')]))
Rnn = ai.RNN(len(UniqueWords), 2, "ManyToOne")

for Generation in range(100000):
  Random = np.random.randint(0, len(Comments))

  Input = np.zeros((len(Ai.WordToBinary(Comments[Random])), 8))
  for i in range(len(Ai.WordToBinary(Comments[Random]))):
    Input[i, :] = np.array(list(Ai.WordToBinary(Comments[Random])[i]))

  Output = Rnn.ForwardProp(Input)

  RealOutput = np.array([1, 0]) if Random < 5331 else np.array([0, 1])

  OutputError = RealOutput - Output

  Rnn.BackProp(OutputError)
  wandb.log({"Error": np.sum(np.abs(OutputError))})