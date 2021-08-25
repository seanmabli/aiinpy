import numpy as np
from emnist import extract_training_samples, extract_test_samples
from TestSrc.NN import NN
from TestSrc.CONV import CONV
from TestSrc.POOL import POOL
from alive_progress import alive_bar
import sys
import wandb

wandb.init(project='cnn')

InToConv1 = CONV((4, 3, 3), LearningRate=0.01, Padding=True, Activation='ReLU')
Conv1ToPool1 = POOL((2, 2))
DenseInToOut = NN(InSize=(4 * 14 * 14), OutSize=10, Activation='StableSoftmax', LearningRate=0.1, WeightsInit=(0, 0))

# Load EMNIST Training An1d Testing Images
TestImageLoaded = 1000
TrainingImages, TrainingLabels = extract_training_samples('digits')
TestImages, TestLabels = extract_test_samples('digits')[0 : TestImageLoaded]

NumOfTrainGen = 5000
with alive_bar(NumOfTrainGen + TestImageLoaded) as bar:
  for Generation in range(NumOfTrainGen):
    # Set Input
    Random = np.random.randint(0, len(TrainingLabels))
    In = (TrainingImages[Random] / 255) - 0.5
    RealOut = np.zeros(10)
    RealOut[TrainingLabels[Random]] = 1
    
    # Forward Propagation
    Conv1 = InToConv1.ForwardProp(In)
    Pool1 = Conv1ToPool1.ForwardProp(Conv1)

    DenseIn = Pool1.flatten()
    Out = DenseInToOut.ForwardProp(DenseIn)

    # Back Propagation
    OutError = RealOut - Out
    DenseInError = DenseInToOut.BackProp(OutError)
    Pool1Error = DenseInError.reshape(Pool1.shape)

    Conv1Error = Conv1ToPool1.BackProp(Pool1Error)
    InError = InToConv1.BackProp(Conv1Error)
    
    wandb.log({"In": np.sum(abs(In)) / In.size,
               "Conv1": np.sum(abs(Conv1)) / Conv1.size,
               "Pool1": np.sum(abs(Pool1)) / Pool1.size,
               "DenseIn": np.sum(abs(DenseIn)) / DenseIn.size, 
               "Out": np.sum(abs(Out)) / Out.size, 

               "OutError": np.sum(abs(OutError)) / OutError.size, 
               "DenseInError": np.sum(abs(DenseInError)) / DenseInError.size, 
               "Pool1Error": np.sum(abs(Pool1Error)) / Pool1Error.size,
               "Conv1Error": np.sum(abs(Conv1Error)) / Conv1Error.size,
               "InError": np.sum(abs(InError)) / InError.size,
               "Correct": 1 if np.argmax(Out) == TrainingLabels[Random] else 0})

    bar()
  
  NumberCorrect = 0
  for Generation in range(TestImageLoaded):
    In = (TestImages[Generation] / 255) - 0.5
    Conv1 = InToConv1.ForwardProp(In)
    Pool1 = Conv1ToPool1.ForwardProp(Conv1)

    DenseIn = Pool1.flatten()
    Out = DenseInToOut.ForwardProp(DenseIn)
    
    NumberCorrect += 1 if np.argmax(Out) == TestLabels[Generation] else 0
    
    wandb.log({"Test Correct": 1 if np.argmax(Out) == TestLabels[Generation] else 0})

    bar()

  print(NumberCorrect / TestImageLoaded)