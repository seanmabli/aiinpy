import numpy as np
from emnist import extract_training_samples, extract_test_samples
import src as ai
# import wandb
from alive_progress import alive_bar

# wandb.init(project="gan-mnist")

# gen -> generator
# gen_model = ai.model(100, (28, 28), [
#   ai.nn(outshape=(128, 7, 7), activation=ai.leakyrelu(0.2), learningrate=0.0002),
#   ai.convtranspose(inshape=(128, 7, 7), filtershape=(128, 4, 4), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
#   ai.convtranspose(inshape=(128, 14, 14), filtershape=(128, 4, 4), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
#   ai.conv(filtershape=(128, 7, 7), learningrate=0.0002, activation=ai.sigmoid(), padding=True)
# ])

# dis -> discrimanator
dismodel = ai.model((28, 28), 1, [
  ai.conv(filtershape=(64, 3, 3), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
  ai.dropout(0.4),
  ai.conv(filtershape=(64, 3, 3), learningrate=0.0002, activation=ai.leakyrelu(0.2), padding=True, stride=(2, 2)),
  ai.dropout(0.4),
  ai.nn(outshape=1, activation=ai.sigmoid(), learningrate=0.0002)
])

# Train Discrimanator
disrealtrain, _ = extract_training_samples('digits')
disrealtest, _ = extract_test_samples('digits')
disrealtrain, disrealtest = disrealtrain[:10000] / 255, disrealtest[:10000] / 255
disfake = np.random.uniform(-0.5, 0.5, disrealtrain.shape)

disintrain, disouttrain = np.vstack((disrealtrain, disfake)), np.hstack((np.ones(len(disrealtrain)), np.zeros(len(disfake))))
disintest, disouttest = np.vstack((disrealtest, disfake)), np.hstack((np.ones(len(disrealtest)), np.zeros(len(disfake))))

dismodel.train(data=(disintrain, disouttrain), numofgen=2000)
print(dismodel.test(data=(disintest, disouttest)))
# wandb.log({"discriminator accuracy": dismodel.test(data=(disintest, disouttest))})

# Train Generator
# for i in range(len(dis_model.model)):
#   dis_model.model[i].learningrate = 0
# 
# numofgen = 10000
# for gen in range(numofgen):
#   input = np.random.uniform(-0.5, 0.5, 100)
#   input = gen_model.forward(input)
#   input = np.average(input, axis=0)
#   out = dis_model.forward(input)
#   print(out)
#   gen_model_error = dis_model.backward(1 - out)
#   gen_model_error = np.array([gen_model_error] * 128)
#   gen_model.backward(gen_model_error)
# 
#   wandb.log({"Generator Error": 1 - out})
#   sys.stdout.write('\r generation: ' + str(gen + 1) + '/' + str(numofgen))