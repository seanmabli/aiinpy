import aiinpy as ai
import numpy as np
import inspect
from importlib_metadata import version
import json
import firebase_admin
from firebase_admin import firestore

cred = firebase_admin.credentials.Certificate('adminkey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

classes = np.array([], dtype=object)
source = np.array([])

title = np.array([])
model = np.array([])
discription = []
url = np.array([])
function = np.array([])
sourcecode = np.array([])
description = np.array([])

for name, obj in inspect.getmembers(ai):
  if inspect.isclass(obj):
    classes = np.append(classes, obj)

for i in range(len(classes)):
  # title, url, id, sourcecode
  title = np.append(title, 'aiinpy.' + classes[i].__name__)
  url = np.append(url, '/' + version('aiinpy') + '/' + classes[i].__name__)
  function = np.append(function, classes[i].__name__ )
  sourcecode = np.append(sourcecode, 'https://github.com/seanmabli/aiinpy/blob/' + version('aiinpy') + '/aiinpy/' + classes[i].__name__ + '.py')
  description = np.append(description, '')

  # model
  source = np.append(source, inspect.getsource(classes[i]))
  if source[i].find('__init__') != -1:
    model = np.append(model, 'aiinpy.' + classes[i].__name__ + source[i][source[i].find('__init__') + 8 : source[i].find('):') + 1])
  else:
    model = np.append(model, 'aiinpy.' + classes[i].__name__ + '()')
for i in range(len(title)):
  db.collection('documentation').add({'title' : title.tolist()[i], 'model' : model.tolist()[i], 'description' : description[i], 'url' : url.tolist()[i], 'sourcecode' : sourcecode.tolist()[i], 'function' : function.tolist()[i], 'version' : version('aiinpy')})
