import testsrc as ai
import numpy as np
import inspect
import json

classes = np.array([], dtype=object)
title = np.array([])
source = np.array([])
url = np.array([])

for name, obj in inspect.getmembers(ai):
  if inspect.isclass(obj):
    classes = np.append(classes, obj)

for i in range(len(classes)):
  title = np.append(title, 'aiinpy.' + classes[i].__name__)
  url = np.append(url, classes[i].__name__)
  source = np.append(source, inspect.getsource(classes[i]))
  if source[i].find('__init__') != -1:
    source[i] = 'aiinpy' + classes[i].__name__ + source[i][source[i].find('__init__') + 8 : source[i].find('):') + 1]
  else:
    source[i] = 'aiinpy' + classes[i].__name__ + '()'

data = np.array([title, source, url], dtype=str)
data = np.rot90(data)

with open("website\src\content.json", "w") as write_file:
  json.dump({"pages": data.tolist()}, write_file, sort_keys=True, indent=2)