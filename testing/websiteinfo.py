import aiinpy as ai
import numpy as np
import inspect
import json

classes = np.array([], dtype=object)
source = np.array([])

title = np.array([])
model = np.array([])
url = np.array([])
urlid = np.array([])

for name, obj in inspect.getmembers(ai):
  if inspect.isclass(obj):
    classes = np.append(classes, obj)

for i in range(len(classes)):
  title = np.append(title, 'aiinpy.' + classes[i].__name__)
  url = np.append(url, '/' + classes[i].__name__)
  urlid = np.append(urlid, classes[i].__name__ )
  source = np.append(source, inspect.getsource(classes[i]))
  if source[i].find('__init__') != -1:
    model = np.append(model, 'aiinpy.' + classes[i].__name__ + source[i][source[i].find('__init__') + 8 : source[i].find('):') + 1])
  else:
    model = np.append(model, 'aiinpy.' + classes[i].__name__ + '()')

data = np.array([title, model, url, urlid], dtype=str)
data = np.rot90(data).tolist()

jsondata = [{"title" : data[i][0], "model" : data[i][1], "url" : data[i][2], "id" : data[i][3]} for i in range(len(data))]

with open("website\src\content.json", "w") as write_file:
  json.dump(jsondata, write_file, indent=2)