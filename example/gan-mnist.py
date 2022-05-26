import numpy as np
from emnist import extract_training_samples, extract_test_samples
import src as ai
# import wandb
import sys

# wandb.init(project="gan-mnist")

# gen -> generator
genmodel = ai.model(inshape=100, outshape=(28, 28), _model=[
  ai.nn(outshape=(128, 7, 7), activation=ai.leakyrelu(0.2), learningrate=0.0002),
  ai.convtranspose(inshape=(128, 7, 7), filtershape=(128, 4, 4), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
  ai.convtranspose(inshape=(128, 14, 14), filtershape=(128, 4, 4), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
  ai.conv(filtershape=(128, 7, 7), learningrate=0.0002, activation=ai.sigmoid(), padding=True),
  ai.average()
])

# dis -> discrimanator
dismodel = ai.model(inshape=(28, 28), outshape=1, _model=[
  ai.conv(filtershape=(64, 3, 3), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
  ai.dropout(0.4),
  ai.conv(filtershape=(64, 3, 3), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
  ai.dropout(0.4),
  ai.nn(outshape=1, activation=ai.sigmoid(), learningrate=0.0002)
], usebestcache=False)

combomodel = ai.model(inshape=100, outshape=1, _model=[
  genmodel,
  dismodel
])

# Train Discrimanator
disrealtrain, _ = extract_training_samples('digits')
disrealtest, _ = extract_test_samples('digits')
disrealtrain, disrealtest = disrealtrain[:10000] / 255, disrealtest[:2000] / 255
disfaketrain, disfaketest = np.random.uniform(-0.5, 0.5, disrealtrain.shape), np.random.uniform(-0.5, 0.5, disrealtest.shape)

disintrain, disouttrain = np.vstack((disrealtrain, disrealtrain)), np.hstack((np.ones(len(disrealtrain)), np.zeros(len(disrealtrain))))
disintest, disouttest = np.vstack((disrealtest, disfaketest)), np.hstack((np.ones(len(disrealtest)), np.zeros(len(disfaketest))))

print('train discrimanator')
dismodel.train(data=(disintrain, disouttrain), numofgen=2000)
# print('discrimanator accuracy:', dismodel.test(data=(disintest, disouttest)))
# wandb.log({"discriminator accuracy": dismodel.test(data=(disintest, disouttest))})

# Train Generator
for i in range(len(dismodel.model)):
  dismodel.model[i].learningrate = 0

numofgen = 3
for gen in range(numofgen):
  input = np.random.uniform(-0.5, 0.5, 100)
  out = combomodel.forward(input)
  error = combomodel.backward(1 - out)

  # wandb.log({"Generator Error": 1 - out})
  sys.stdout.write('\r' + 'generation: ' + str(gen + 1) + '/' + str(numofgen))