import numpy as np
from emnist import extract_training_samples, extract_test_samples
import testsrc as ai
# import wandb

# wandb.init(project="gan-mnist")

# gen -> generator
gen_model = ai.model(100, (28, 28), [
  ai.nn(100, (128, 7, 7), 0.0002, ai.leakyrelu(0.2)),
  ai.convtranspose((128, 7, 7), (128, 4, 4), 0.0002, ai.leakyrelu(0.2), True, (2, 2)),
  ai.convtranspose((128, 14, 14), (128, 4, 4), 0.0002, ai.leakyrelu(0.2), True, (2, 2)),
  ai.nn((128, 28, 28), (28, 28), ai.sigmoid(), 0.0002)
])

# dis -> discrimanator
dis_model = ai.model((28, 28), 1, [
  ai.conv((28, 28), (64, 3, 3), 0.0002, ai.leakyrelu(0.2), True, (2, 2)),
  ai.conv((64, 14, 14), (64, 3, 3), 0.0002, ai.leakyrelu(0.2), True, (2, 2)),
  ai.nn((64, 7, 7), 1, ai.sigmoid(), 0.0002)
])

# Train Discrimanator
DisRealData, _ = extract_training_samples('digits')
DisRealData = (DisRealData[0 : 100000] / 255) - 0.5
DisFakeData = np.random.uniform(-0.5, 0.5, DisRealData.shape)
DisDataIn, DisDataOut = np.vstack((DisRealData, DisFakeData)), np.hstack((np.ones(len(DisRealData)), np.zeros(len(DisFakeData))))

dis_model.train(DisDataIn, DisDataOut, 2000)
print(dis_model.test(TestDataIn, TestDataOut))

'''
for i in range(len(dis_model)):
  dis_model.Model[i].LearningRate = 0

for Generation in range(10000):
  In = np.zeros(10)
  In[np.random.randint(0, 10)] = 1
  In = gen_model.forward(In)
  Out = dis_model.forward(In)
  gen_model_error = dis_model.backward(1 - Out)
  gen_model.backprop(gen_model_error)
'''