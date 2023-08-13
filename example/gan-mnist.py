import numpy as np
from emnist import extract_training_samples, extract_test_samples
import src as ai
import sys, time

# gen -> generator
genmodel = ai.model(inshape=100, outshape=(28, 28), layers=[
  ai.nn(outshape=(128, 7, 7), activation=ai.leakyrelu(0.2), learningrate=0.0002),
  ai.convtranspose(inshape=(128, 7, 7), filtershape=(128, 4, 4), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
  ai.convtranspose(inshape=(128, 14, 14), filtershape=(128, 4, 4), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
  ai.conv(numoffilters=128, filtershape=(7, 7), learningrate=0.0002, activation=ai.sigmoid(), padding=True),
  ai.mean()
])

# dis -> discrimanator
dismodel = ai.model(inshape=(28, 28), outshape=1, layers=[
  ai.conv(numoffilters=64, filtershape=(3, 3), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2), clip=True),
  ai.dropout(0.4),
  ai.conv(numoffilters=64, filtershape=(3, 3), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2), clip=True),
  ai.dropout(0.4),
  ai.nn(outshape=1, activation=ai.sigmoid(), learningrate=0.0002, clip=True)
])

# combomodel = ai.model(inshape=100, outshape=1, layers=[
#   genmodel,
#   dismodel
# ])

# Train Discrimanator
disrealtrain, _ = extract_training_samples('digits')
disrealtest, _ = extract_test_samples('digits')
disrealtrain, disrealtest = disrealtrain[:10000] / 255, disrealtest[:2000] / 255
disfaketrain, disfaketest = np.random.uniform(-0.5, 0.5, disrealtrain.shape), np.random.uniform(-0.5, 0.5, disrealtest.shape)

disintrain, disouttrain = np.vstack((disrealtrain, disrealtrain)), np.hstack((np.ones(len(disrealtrain)), np.zeros(len(disrealtrain))))
disintest, disouttest = np.vstack((disrealtest, disfaketest)), np.hstack((np.ones(len(disrealtest)), np.zeros(len(disfaketest))))

print('train discrimanator')
dismodel.train(data=(disintrain, disouttrain), numofgen=2000)
print('discrimanator accuracy:', dismodel.test(data=(disintest, disouttest)))

for layer in dismodel.layers:
  layer.learningrate = 0

numofgen = 100000
avgtime = []
trainstarttime = time.time()

for gen in range(numofgen):
  starttime = time.time()
  input = np.random.uniform(-0.5, 0.5, 100)
  out = genmodel.forward(input)
  out = dismodel.forward(out)
  error = dismodel.backward(1 - out)
  genmodel.backward(error)

  avgtime.append(time.time() - starttime)
  speed = round(6000 / sum(avgtime[-100:]))
  sys.stdout.write('\r' + 'training: ' + str(gen + 1) + '/' + str(numofgen) + ' | ' + str(speed) + ' gen/min | ' + str(round(60 * (numofgen - gen) / speed)) + ' sec remaining | ' + str(round(time.time() - trainstarttime)) + ' sec elapsed')

  if gen % 1000 == 0:
    for _ in range(3):
      input = np.random.uniform(-0.5, 0.5, 100)
      out = genmodel.forward(input)