import testsrc as ai
import numpy as np
import sys
import inspect

files = np.array([], dtype=str)
classes = np.array([], dtype=str)
inits = np.array([], dtype=str)
source = inspect.getsource(ai)

while source.find('from') != -1:
  files = np.append(files, source[source.find('from') + 6 : source.find('import') - 1])
  source = source[source.find('import') + 6:]

for i in range(len(files)):
  source = inspect.getsource(getattr(ai, files[i]))
  while source.find('from') != -1:
    files = np.append(files, source[source.find('from') + 6 : source.find('import') - 1])
    source = source[source.find('import') + 6:]
    i = 0
  while source.find('class') != -1:
    if source[source.find('class') + 6 : source.find(':')] != '':
      classes = np.append(classes, source[source.find('class') + 6 : source.find(':')])
    source = source[source.find(':') + 1:]
    i = 0
  while source.find('init') != -1:
    inits = np.append(inits, source[source.find('__(') + 2 : source.find(':')])
    source = source[source.find(')') + 1:]
    i = 0

print(files)
print(classes)
print(inits)