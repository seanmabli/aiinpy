from __future__ import annotations
import numpy as np
import sys, os, time, json, random, datetime
import _pickle as pickle
import wandb
'''
import pyrebase

firebase = pyrebase.initialize_app({
  "apiKey" : "AIzaSyAeVOKLeGPek386fDsyZ7lC9zQ_9JlnaIc",
  "authDomain" : "aiinpy.firebaseapp.com",
  "databaseURL": "https://aiinpy-default-rtdb.firebaseio.com/",
  "storageBucket" : "aiinpy.appspot.com"
})
db = firebase.database()

def error(func):
    def inner(*args, **kwargs):
      try:
        func(*args, **kwargs)
      except Exception as e:
        metadata = open([x[0] + '/metadata.json' for x in os.walk('aiinpy')][1], 'r').read()
        # db.child("errors").push({"error" : str(e), 'metadata' : metadata})
    return inner

@error
'''
class model:
  def __init__(self, inshape, outshape, _model, wandbproject=None, usebestcache=False, usespecificcache='', cacheexpire=10):
    self.inshape = inshape = inshape if isinstance(inshape, tuple) else tuple([inshape])
    self.outshape = outshape if isinstance(outshape, tuple) else tuple([outshape])
    self._model = _model
    self.wandbproject = wandbproject

    i = 0
    while i < len(self._model):
      if isinstance(self._model[i], model):
        c = self._model[i]
        for j, newlayer in enumerate(self._model[i]._model):
          self._model.insert(i + j, newlayer)
        self._model.remove(c)
        i = -1
      i += 1

    for layer in self._model:
      inshape = layer.modelinit(inshape)
      print(inshape)
    
    if wandbproject is not None:
      wandb.init(project=wandbproject)

    print_model = [p.__repr__() for p in self._model]

    lowesterror = float('inf')
    if usebestcache and os.path.isdir('aiinpy'):
      for run in [x[0] + '/metadata.json' for x in os.walk('aiinpy')][1:]:
        info = json.load(open(run, 'r'))
        if print_model == info['_model'] and info['cacheexpire'] > 0:
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
      self._model = pickle.load(open('aiinpy/' + bestcache + '/_model.pickle', 'rb'))
      # except:
      #   print('no cache available')

    self.time = datetime.datetime.now()
    self.runname = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
    self.longrunname = self.time.strftime('%m%d%Y%H%M%S') + '-' + self.runname
    if not os.path.isdir('aiinpy'):
      os.mkdir('aiinpy')
    else:
      for run in [x[0] + '/metadata.json' for x in os.walk('aiinpy')][1:]:
        info = json.load(open(run, 'r'))
        info['cacheexpire'] -= 1 if info['cacheexpire'] > 0 else 0
        json.dump(info, open(run, 'w'), indent=2)
    os.mkdir('aiinpy/' + self.longrunname)
    json.dump({'name' : self.runname, 'file' : self.longrunname, 'time' : str(self.time), '_model' : print_model, 'cacheexpire' : cacheexpire}, open('aiinpy/' + self.longrunname + '/metadata.json', 'w'), indent=2)

  def forward(self, input):
    for i in range(len(self._model)):
      input = self._model[i].forward(input)
    return input

  def backward(self, outError):
    for i in reversed(range(len(self._model))):
      outError = self._model[i].backward(outError)
    return outError

  def train(self, data, numofgen):
    # data preprocessing: tuple of (indata, outdata) with indata and outdata as numpy array
    data = list(data) if type(data) == tuple else data
    data[0] = np.reshape(data[0], (data[0].shape[0], 1)) if len(data[0].shape) == 1 else data[0]
    data[1] = np.reshape(data[1], (data[1].shape[0], 1)) if len(data[1].shape) == 1 else data[1]

    NumOfData = (set(self.inshape) ^ set(data[0].shape)).pop()
    if data[0].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[0].shape)))
      x.pop(data[0].shape.index(NumOfData))
      data[0] = np.transpose(data[0], tuple([data[0].shape.index(NumOfData)]) + tuple(x))
    if data[1].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[1].shape)))
      x.pop(data[1].shape.index(NumOfData))
      data[1] = np.transpose(data[1], tuple([data[1].shape.index(NumOfData)]) + tuple(x))

    trainerror = []

    # Training, with wandb
    if self.wandbproject is not None:
      for gen in range(numofgen):
        random = np.random.randint(0, NumOfData)
        input = data[0][random]
        for i in range(len(self._model)):
          input = self._model[i].forward(input)

        outerror = data[1][random] - input
        wandb.log({'train error': np.sum(abs(outerror))})
        trainerror.append(np.sum(abs(outerror)))
        for i in reversed(range(len(self._model))):
          outerror = self._model[i].backward(outerror)
        sys.stdout.write('\r' + 'training: ' + str(gen + 1) + '/' + str(numofgen))
    else:
      # Training, without wandb
      for gen in range(numofgen):
        random = np.random.randint(0, NumOfData)
        input = data[0][random]
        for i in range(len(self._model)):
          input = self._model[i].forward(input)

        outerror = data[1][random] - input
        trainerror.append(np.sum(abs(outerror)))
        for i in reversed(range(len(self._model))):
          outerror = self._model[i].backward(outerror)
        sys.stdout.write('\r' + 'training: ' + str(gen + 1) + '/' + str(numofgen))
    
    print('')

    pickle.dump(self._model, open('aiinpy/' + self.longrunname + '/_model.pickle', 'wb'))
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
    data[0] = np.reshape(data[0], (data[0].shape[0], 1)) if len(data[0].shape) == 1 else data[0]
    data[1] = np.reshape(data[1], (data[1].shape[0], 1)) if len(data[1].shape) == 1 else data[1]

    NumOfData = (set(self.inshape) ^ set(data[0].shape)).pop()
    if data[0].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[0].shape)))
      x.pop(data[0].shape.index(NumOfData))
      data[0] = np.transpose(data[0], tuple([data[0].shape.index(NumOfData)]) + tuple(x))
    if data[1].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[1].shape)))
      x.pop(data[1].shape.index(NumOfData))
      data[1] = np.transpose(data[1], tuple([data[1].shape.index(NumOfData)]) + tuple(x))

    testerror = []

    # Testing
    testcorrect = 0
    for gen in range(NumOfData):
      input = data[0][gen]
      for i in range(len(self._model)):
        input = self._model[i].forward(input)

      testerror.append(np.sum(abs(data[1][gen] - input)))
      testcorrect += 1 if np.argmax(input) == np.argmax(data[1][gen]) else 0
      sys.stdout.write('\r' + 'testing: ' + str(gen + 1) + '/' + str(NumOfData))

    print('')
    
    testaccuracy = testcorrect / NumOfData

    if self.wandbproject is not None:
      wandb.log({'test accuracy': testaccuracy})

    info = json.load(open('aiinpy/' + self.longrunname + '/metadata.json', 'r'))
    info.update({'testerror' : sum(testerror) / NumOfData, 'testaccuracy' : testaccuracy})
    json.dump(info, open('aiinpy/' + self.longrunname + '/metadata.json', 'w'), indent=2)

    return testaccuracy

  def use(self, indata):
    indata = np.reshape(indata, (indata.shape[0], 1)) if len(indata.shape) == 1 else indata
    NumOfData = (set(self.inshape) ^ set(indata.shape)).pop()
    if indata.shape.index(NumOfData) != 0:
      x = list(range(0, len(indata.shape)))
      x.pop(indata.shape.index(NumOfData))
      indata = np.transpose(indata, tuple([indata.shape.index(NumOfData)]) + tuple(x))

    outdata = np.zeros(indata.shape)
    for gen in range (NumOfData):
      input = indata[gen]
      for i in range(len(self._model)):
        input = self._model[i].forward(input)
      outdata[gen] = input
    return outdata