import numpy as np
from TestSrc.NN import NN
from alive_progress import alive_bar
import wandb

wandb.init(project='nn_weight', entity='sean_m')

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
    Hidden1 = InputToHidden1.forwardprop(Input)
    Hidden2 = Hidden1ToHidden2.forwardprop(Hidden1)
    Output = Hidden2ToOutput.forwardprop(Hidden2)

    # Back Prop
    OutputError = RealOutput - Output
    Hidden2Error = Hidden2ToOutput.backprop(OutputError)
    Hidden1Error = Hidden1ToHidden2.backprop(Hidden2Error)
    InputToHidden1.backprop(Hidden1Error)

    bar()

  # Testing
  NumberCorrect = 0

  for Generation in range(NumOfTestGen):
    # Create Input
    Input = np.random.choice(([0, 1]), 2)
    RealOutput = [1, 0] if sum(Input) == 1 else [0, 1]

    # Forward Prop
    Hidden1 = InputToHidden1.forwardprop(Input)
    Hidden2 = Hidden1ToHidden2.forwardprop(Hidden1)
    Output = Hidden2ToOutput.forwardprop(Hidden2)

    NumberCorrect += 1 if np.argmax(Output) == (0 if sum(Input) == 1 else 1) else 0
    bar()
wandb.log({"InputToHidden1 Weights": InputToHidden1.Weights})
wandb.log({"Hidden1ToHidden2 Weights": Hidden1ToHidden2.Weights})
wandb.log({"Hidden2ToOutput Weights": Hidden2ToOutput.Weights})
wandb.log({"InputToHidden1 Biases": InputToHidden1.Biases})
wandb.log({"Hidden1ToHidden2 Biases": Hidden1ToHidden2.Biases})
wandb.log({"Hidden2ToOutput Biases": Hidden2ToOutput.Biases})
wandb.log({"Accuracy": (NumberCorrect / NumOfTestGen)})