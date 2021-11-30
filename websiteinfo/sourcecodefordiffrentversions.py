import subprocess
import json
import pickle

possibleversions = [2, 3, 7, 10, 11, 13, 15, 16]
data = []

for version in possibleversions:
  subprocess.run('pip install aiinpy==0.0.' + str(version))
  subprocess.run('python3 websiteinfo/websiteinfo.py')
  data += pickle.load(open('websiteinfo/datatransfer.txt', 'rb'))
  
json.dump(data, open("website\src\content.json", "w"), indent=2)