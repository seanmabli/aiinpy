import numpy as np
from NN import NN

# Neural Network Model
InputToHidden1 = NN(InputSize=2, OutputSize=16, Activation='ReLU', LearningRate=0.1, DropoutRate=0)
Hidden1ToHidden2 = NN(InputSize=16, OutputSize=16, Activation='ReLU', LearningRate=0.1, DropoutRate=0)
Hidden2ToOutput = NN(InputSize=16, OutputSize=2, Activation='Sigmoid', LearningRate=0.1, DropoutRate=0)

# Training
for Generation in range(100):
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

# Testing
NumberCorrect = 0

for Generation in range(100):
  # Create Input
  Input = np.random.choice(([0, 1]), 2)
  RealOutput = [1, 0] if sum(Input) == 1 else [0, 1]
  
  # Forward Prop
  Hidden1 = InputToHidden1.ForwardProp(Input)
  Hidden2 = Hidden1ToHidden2.ForwardProp(Hidden1)
  Output = Hidden2ToOutput.ForwardProp(Hidden2)
  
  NumberCorrect += 1 if np.argmax(Output) == (0 if sum(Input) == 1 else 1) else 0

print(NumberCorrect / 100)