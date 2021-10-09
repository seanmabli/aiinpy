import numpy as np
from TestSrc.NN import NN
from alive_progress import alive_bar

# Neural Network Model
InputToHidden1 = NN(InShape=2, OutShape=16, Activation='ReLU', LearningRate=0.1)
Hidden1ToHidden2 = NN(InShape=16, OutShape=16, Activation='ReLU', LearningRate=0.1)
Hidden2ToOutput = NN(InShape=16, OutShape=2, Activation='Sigmoid', LearningRate=0.1)

# Training
NumOfTrainGen = 100
NumOfTestGen = 100

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):
    # Create Input And Real Output
    Input = np.random.choice(([0, 1]), 2)
    RealOutput = [1, 0] if sum(Input) == 1 else [0, 1]

    # Forward Prop
    Hidden1 = InputToHidden1.ForwardProp(Input)
    Hidden2 = Hidden1ToHidden2.ForwardProp(Hidden1)
    Output = Hidden2ToOutput.ForwardProp(Hidden2)

    # Back Prop
    OutputError = RealOutput - Output
    Hidden2Error = Hidden2ToOutput.BackProp(OutputError)
    Hidden1Error = Hidden1ToHidden2.BackProp(Hidden2Error)
    InputToHidden1.BackProp(Hidden1Error)

    bar()

  # Testing
  NumberCorrect = 0

  for Generation in range(NumOfTestGen):
    # Create Input
    Input = np.random.choice(([0, 1]), 2)
    RealOutput = [1, 0] if sum(Input) == 1 else [0, 1]

    # Forward Prop
    Hidden1 = InputToHidden1.ForwardProp(Input)
    Hidden2 = Hidden1ToHidden2.ForwardProp(Hidden1)
    Output = Hidden2ToOutput.ForwardProp(Hidden2)

    NumberCorrect += 1 if np.argmax(Output) == (0 if sum(Input) == 1 else 1) else 0
    bar()

  print(NumberCorrect / NumOfTestGen)