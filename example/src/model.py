from __future__ import annotations
import sys, os, time, json, random, datetime
import _pickle as pickle
import wandb
from .tensor import tensor

class model:
  def __init__(self, inshape, outshape, layers, wandbproject=None, usebestcache=False, usespecificcache='', cacheexpire=10):
    self.inshape = inshape = inshape if isinstance(inshape, tuple) else tuple([inshape])
    self.outshape = outshape if isinstance(outshape, tuple) else tuple([outshape])
    self.layers = layers
    self.wandbproject = wandbproject

    i = 0
    while i < len(self.layers):
      if isinstance(self.layers[i], model):
        c = self.layers[i]
        for j, newlayer in enumerate(self.layers[i].layers):
          self.layers.insert(i + j, newlayer)
        self.layers.remove(c)
        i = -1
      i += 1

    for layer in self.layers:
      inshape = layer.modelinit(inshape)
    
    if wandbproject is not None:
      wandb.init(project=wandbproject)

    printlayers = [p.__repr__() for p in self.layers]

    lowesterror = float('inf')
    if usebestcache and os.path.isdir('aiinpy'):
      for run in [x[0] + '/metadata.json' for x in os.walk('aiinpy')][1:]:
        info = json.load(open(run, 'r'))
        if printlayers == info['layers'] and info['cacheexpire'] > 0:
          try:
            try:
              if info['testerror'] < lowesterror:
                bestcache = info['file']
                lowesterror = info['testerror']
            except:
              if info['trainerror'] < lowesterror:
                bestcache = info['file']
                lowesterror = info['trainerror']
          except:
            pass

      # try:
      self.layers = pickle.load(open('aiinpy/' + bestcache + '/layers.pickle', 'rb'))
      # except:
      #   print('no cache available')

    self.time = datetime.datetime.now()
    self.runname = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
    self.longrunname = self.time.strftime('%m%d%Y%H%M%S') + '-' + self.runname
    if not os.path.isdir('aiinpy'):
      os.mkdir('aiinpy')
    else:
      for run in [x[0] + '/metadata.json' for x in os.walk('aiinpy')][1:]:
        try:
          info = json.load(open(run, 'r'))
          info['cacheexpire'] -= 1 if info['cacheexpire'] > 0 else 0
          json.dump(info, open(run, 'w'), indent=2)
        except:
          if input('cache', run, 'is corrupted, do you want to delete it (y/n)? ').lower() == 'y':
            os.remove(run)
            os.remove(run.replace('metadata.json', 'layers.pickle'))
            os.remove(run.replace('metadata.json', 'trainerror.pickle'))
            os.remove(run.replace('metadata.json', 'testerror.pickle'))
          quit()
    os.mkdir('aiinpy/' + self.longrunname)
    json.dump({'name' : self.runname, 'file' : self.longrunname, 'time' : str(self.time), 'layers' : printlayers, 'cacheexpire' : cacheexpire}, open('aiinpy/' + self.longrunname + '/metadata.json', 'w'), indent=2)

  def __repr__(self):
    return "\n".join([p.__repr__() for p in self.layers])

  def forward(self, input):
    for i in range(len(self.layers)):
      input = self.layers[i].forward(input)
    return input

  def backward(self, outError):
    for i in reversed(range(len(self.layers))):
      outError = self.layers[i].backward(outError)
    return outError

  def train(self, data, numofgen):
    # data preprocessing: tuple of (indata, outdata) with indata and outdata as numpy array
    data = list(data) if type(data) == tuple else data
    data[0] = tensor.reshape(data[0], (data[0].shape[0], 1)) if len(data[0].shape) == 1 else data[0]
    data[1] = tensor.reshape(data[1], (data[1].shape[0], 1)) if len(data[1].shape) == 1 else data[1]

    NumOfData = (set(self.inshape) ^ set(data[0].shape)).pop()
    if data[0].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[0].shape)))
      x.pop(data[0].shape.index(NumOfData))
      data[0] = tensor.transpose(data[0], tuple([data[0].shape.index(NumOfData)]) + tuple(x))
    if data[1].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[1].shape)))
      x.pop(data[1].shape.index(NumOfData))
      data[1] = tensor.transpose(data[1], tuple([data[1].shape.index(NumOfData)]) + tuple(x))

    trainerror = []
    avgtime = []
    trainstarttime = time.time()

    # Training, with wandb
    if self.wandbproject != None:
      for gen in range(numofgen):
        starttime = time.time()
        random = int(tensor.uniform() * NumOfData)
        input = data[0][random]
        for i in range(len(self.layers)):
          input = self.layers[i].forward(input)

        outerror = data[1][random] - input
        wandb.log({'train error': tensor.sum(abs(outerror))})
        trainerror.append(tensor.sum(abs(outerror)))
        for i in reversed(range(len(self.layers))):
          outerror = self.layers[i].backward(outerror)
        avgtime.append(time.time() - starttime)
        speed = round(6000 / sum(avgtime[-100:]))
        remaining = round(60 * (numofgen - gen) / speed)
        remaining = f"{(remaining // 3600):02}" + ':' + f"{((remaining % 3600) // 60):02}" + ':' + f"{(remaining % 60):02}"
        elapsed = round(time.time() - trainstarttime)
        elapsed = f"{(elapsed // 3600):02}" + ':' + f"{((elapsed % 3600) // 60):02}" + ':' + f"{(elapsed % 60):02}"
        sys.stdout.write('\r' + 'training: ' + str(gen + 1) + '/' + str(numofgen) + ' | ' + str(speed) + ' gen/min | ' + remaining + ' remaining | ' + elapsed + ' elapsed')
    else:
      # Training, without wandb
      for gen in range(numofgen):
        starttime = time.time()
        random = int(tensor.uniform() * NumOfData)
        input = data[0][random]
        for i in range(len(self.layers)):
          input = self.layers[i].forward(input)

        outerror = data[1][random] - input
        trainerror.append(tensor.sum(abs(outerror)))
        for i in reversed(range(len(self.layers))):
          outerror = self.layers[i].backward(outerror)
        avgtime.append(time.time() - starttime)
        speed = round(6000 / sum(avgtime[-100:]))
        remaining = round(60 * (numofgen - gen) / speed)
        remaining = f"{(remaining // 3600):02}" + ':' + f"{((remaining % 3600) // 60):02}" + ':' + f"{(remaining % 60):02}"
        elapsed = round(time.time() - trainstarttime)
        elapsed = f"{(elapsed // 3600):02}" + ':' + f"{((elapsed % 3600) // 60):02}" + ':' + f"{(elapsed % 60):02}"
        sys.stdout.write('\r' + 'training: ' + str(gen + 1) + '/' + str(numofgen) + ' | ' + str(speed) + ' gen/min | ' + remaining + ' remaining | ' + elapsed + ' elapsed')

    sys.stdout.write('\r' + 'training: ' + str(gen + 1) + '/' + str(numofgen) + ' | ' + str(speed) + ' gen/min | ' + elapsed + ' elapsed')
    print('')

    pickle.dump(self.layers, open('aiinpy/' + self.longrunname + '/layers.pickle', 'wb'))
    pickle.dump(trainerror, open('aiinpy/' + self.longrunname + '/trainerror.pickle', 'wb'))

    if numofgen * 0.01 > 5:
      simptrainerror = sum(trainerror[- int(numofgen * 0.01) :]) / int(numofgen * 0.01)
    else:
      simptrainerror = sum(trainerror[-5:]) / 5

    info = json.load(open('aiinpy/' + self.longrunname + '/metadata.json', 'r'))
    info.update({'trainerror' : simptrainerror})
    json.dump(info, open('aiinpy/' + self.longrunname + '/metadata.json', 'w'), indent=2)

    return trainerror

  def test(self, data):
    # data preprocessing: tuple of (indata, outdata) with indata and outdata as numpy array
    data = list(data) if type(data) == tuple else data
    data[0] = tensor.reshape(data[0], (data[0].shape[0], 1)) if len(data[0].shape) == 1 else data[0]
    data[1] = tensor.reshape(data[1], (data[1].shape[0], 1)) if len(data[1].shape) == 1 else data[1]

    NumOfData = (set(self.inshape) ^ set(data[0].shape)).pop()
    if data[0].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[0].shape)))
      x.pop(data[0].shape.index(NumOfData))
      data[0] = tensor.transpose(data[0], tuple([data[0].shape.index(NumOfData)]) + tuple(x))
    if data[1].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[1].shape)))
      x.pop(data[1].shape.index(NumOfData))
      data[1] = tensor.transpose(data[1], tuple([data[1].shape.index(NumOfData)]) + tuple(x))

    testerror = []
    avgtime = []
    trainstarttime = time.time()

    # Testing
    testcorrect = 0
    for gen in range(NumOfData):
      starttime = time.time()

      input = data[0][gen]
      for i in range(len(self.layers)):
        input = self.layers[i].forward(input)

      testerror.append(tensor.sum(abs(data[1][gen] - input)))
      testcorrect += 1 if tensor.argmax(input) == tensor.argmax(data[1][gen]) else 0

      avgtime.append(time.time() - starttime)
      speed = round(6000 / sum(avgtime[-100:]))
      remaining = round(60 * (NumOfData - gen) / speed)
      remaining = f"{(remaining // 3600):02}" + ':' + f"{((remaining % 3600) // 60):02}" + ':' + f"{(remaining % 60):02}"
      elapsed = round(time.time() - trainstarttime)
      elapsed = f"{(elapsed // 3600):02}" + ':' + f"{((elapsed % 3600) // 60):02}" + ':' + f"{(elapsed % 60):02}"
      sys.stdout.write('\r' + 'testing: ' + str(gen + 1) + '/' + str(NumOfData) + ' | ' + str(speed) + ' gen/min | ' + remaining + ' remaining | ' + elapsed + ' elapsed')

    sys.stdout.write('\r' + 'testing: ' + str(gen + 1) + '/' + str(NumOfData) + ' | ' + str(speed) + ' gen/min | ' + elapsed + ' elapsed')
    print('')
    
    testaccuracy = testcorrect / NumOfData

    if self.wandbproject is not None:
      wandb.log({'test accuracy': testaccuracy})

    info = json.load(open('aiinpy/' + self.longrunname + '/metadata.json', 'r'))
    info.update({'testerror' : sum(testerror) / NumOfData, 'testaccuracy' : testaccuracy})
    json.dump(info, open('aiinpy/' + self.longrunname + '/metadata.json', 'w'), indent=2)

    return testaccuracy

  def use(self, indata):
    indata = tensor.reshape(indata, (indata.shape[0], 1)) if len(indata.shape) == 1 else indata
    NumOfData = (set(self.inshape) ^ set(indata.shape)).pop()
    if indata.shape.index(NumOfData) != 0:
      x = list(range(0, len(indata.shape)))
      x.pop(indata.shape.index(NumOfData))
      indata = tensor.transpose(indata, tuple([indata.shape.index(NumOfData)]) + tuple(x))

    outdata = tensor.zeros(indata.shape)
    for gen in range (NumOfData):
      input = indata[gen]
      for i in range(len(self.layers)):
        input = self.layers[i].forward(input)
      outdata[gen] = input
    return outdata