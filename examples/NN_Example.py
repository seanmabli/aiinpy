import numpy as np
import aiinpy as ai
import wandb

wandb.init(project='nn')

config = wandb.config
config.InputToHidden1LearningRate = 0.584
config.Hidden1ToHidden2LearningRate = 0.4461
config.Hidden2ToOutputLearningRate = 0.7103
config.Hidden1Length = 73
config.Hidden2Length = 25
config.InputToHidden1Activation = "Tanh"
config.Hidden1ToHidden2Activation = "ReLU"
config.Hidden2ToOutputActivation = "StableSoftMax"
config.NumberOfGenerations = 1000

InputToHidden1 = ai.NN(2, config.Hidden1Length, config.InputToHidden1Activation, config.InputToHidden1LearningRate)
Hidden1ToHidden2 = ai.NN(config.Hidden1Length, config.Hidden2Length, config.Hidden1ToHidden2Activation, config.Hidden1ToHidden2LearningRate)
Hidden2ToOutput = ai.NN(config.Hidden2Length, 2, config.Hidden2ToOutputActivation, config.Hidden2ToOutputLearningRate)
NumberCorrect = 0

for Generation in range(config.NumberOfGenerations):
  Input = np.random.choice(([0, 1]), 2)
  RealOutput = [1, 0] if sum(Input) == 1 else [0, 1]
  
  Hidden1 = InputToHidden1.ForwardProp(Input)
  Hidden2 = Hidden1ToHidden2.ForwardProp(Hidden1)
  Output = Hidden2ToOutput.ForwardProp(Hidden2)
  
  NumberCorrect += 1 if np.argmax(Output) == (0 if sum(Input) == 1 else 1) else 0
  
  OutputError = RealOutput - Output
  Hidden2Error = Hidden2ToOutput.BackProp(OutputError)
  Hidden1Error = Hidden1ToHidden2.BackProp(Hidden2Error)
  InputToHidden1.BackProp(Hidden1Error)

TestAcc = 0

Input = [0, 0]
Hidden1 = InputToHidden1.ForwardProp(Input)
Hidden2 = Hidden1ToHidden2.ForwardProp(Hidden1)
Output = Hidden2ToOutput.ForwardProp(Hidden2)
TestAcc += 1 if np.argmax(Output) == (0 if sum(Input) == 1 else 1) else 0

Input = [1, 0]
Hidden1 = InputToHidden1.ForwardProp(Input)
Hidden2 = Hidden1ToHidden2.ForwardProp(Hidden1)
Output = Hidden2ToOutput.ForwardProp(Hidden2)
TestAcc += 1 if np.argmax(Output) == (0 if sum(Input) == 1 else 1) else 0

Input = [0, 1]
Hidden1 = InputToHidden1.ForwardProp(Input)
Hidden2 = Hidden1ToHidden2.ForwardProp(Hidden1)
Output = Hidden2ToOutput.ForwardProp(Hidden2)
TestAcc += 1 if np.argmax(Output) == (0 if sum(Input) == 1 else 1) else 0

Input = [1, 1]
Hidden1 = InputToHidden1.ForwardProp(Input)
Hidden2 = Hidden1ToHidden2.ForwardProp(Hidden1)
Output = Hidden2ToOutput.ForwardProp(Hidden2)
TestAcc += 1 if np.argmax(Output) == (0 if sum(Input) == 1 else 1) else 0

wandb.log({"Accuracy": TestAcc})
print(TestAcc + (NumberCorrect / Generation))