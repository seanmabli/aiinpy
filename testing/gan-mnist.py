import numpy as np
from emnist import extract_training_samples, extract_test_samples
import testsrc as ai
import wandb
from alive_progress import alive_bar

wandb.init(project="gan-mnist")

# gen -> generator
gen_model = ai.model(100, (28, 28), [
  ai.nn(100, (128, 7, 7), ai.leakyrelu(0.2), 0.0002),
  ai.convtranspose((128, 7, 7), (128, 4, 4), 0.0002, ai.leakyrelu(0.2), True, (2, 2)),
  ai.convtranspose((128, 14, 14), (128, 4, 4), 0.0002, ai.leakyrelu(0.2), True, (2, 2)),
  ai.conv((128, 28, 28), (128, 7, 7), 0.0002, ai.sigmoid(), True)
])

# dis -> discrimanator
dis_model = ai.model((28, 28), 1, [
  ai.conv((28, 28), (64, 3, 3), 0.0002, ai.leakyrelu(0.2), True, (2, 2)),
  ai.conv((64, 14, 14), (64, 3, 3), 0.0002, ai.leakyrelu(0.2), True, (2, 2)),
  ai.nn((64, 7, 7), 1, ai.sigmoid(), 0.0002)
])

# Train Discrimanator
DisRealData, _ = extract_training_samples('digits')
DisRealData = (DisRealData[0 : 10000] / 255) - 0.5
DisFakeData = np.random.uniform(-0.5, 0.5, DisRealData.shape)
DisDatain, DisDataout = np.vstack((DisRealData, DisFakeData)), np.hstack((np.ones(len(DisRealData)), np.zeros(len(DisFakeData))))

dis_model.train(DisDatain, DisDataout, 2000)
wandb.log({"Discriminator accuracy": dis_model.test(DisDatain, DisDataout)})

# Train Generator
for i in range(len(dis_model.Model)):
  dis_model.Model[i].learningrate = 0

with alive_bar(10000) as bar:
  for Generation in range(10000):
    in = np.random.uniform(-0.5, 0.5, 100)
    in = gen_model.forward(in)
    in = np.average(in, axis=0)
    out = dis_model.forward(in)
    gen_model_error = dis_model.backward(1 - out)
    gen_model_error = np.array([gen_model_error] * 128)
    gen_model.backward(gen_model_error)

    wandb.log({"Generator Error": 1 - out})
    bar()