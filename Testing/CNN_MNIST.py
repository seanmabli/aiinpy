import numpy as np
from emnist import extract_training_samples, extract_test_samples
from TestSrc.NN import NN
from TestSrc.CONV import CONV
from TestSrc.POOL import POOL
from alive_progress import alive_bar
import wandb

wandb.init(project='pool_test')

InToConv1 = CONV((4, 3, 3), LearningRate=0.01, Padding=True, Activation='ReLU')
Conv1ToPool1 = POOL((2, 2), (2, 2), 'Max')
DenseInToOut = NN(InShape=(4 * 14 * 14), OutShape=10, Activation='StableSoftmax', LearningRate=0.1, WeightsInit=(0, 0))

# Load EMNIST Training And Testing Images
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
    
    wandb.log({"Train Correct": 1 if np.argmax(Out) == TrainingLabels[Random] else 0})

    bar()
  
  NumberCorrect = 0
  for Generation in range(TestImageLoaded):
    In = (TestImages[Generation] / 255) - 0.5
    Conv1 = InToConv1.ForwardProp(In)
    Pool1 = Conv1ToPool1.ForwardProp(Conv1)

    DenseIn = Pool1.flatten()
    Out = DenseInToOut.ForwardProp(DenseIn)
    
    NumberCorrect += 1 if np.argmax(Out) == TestLabels[Generation] else 0
    
    bar()

  wandb.log({"Test Correct": NumberCorrect / TestImageLoaded})
  print(NumberCorrect / TestImageLoaded)