import subprocess
import inspect
import numpy as np
import runpy

classesbyversion = np.array([])

for version in range(17):
  subprocess.run('pip install aiinpy==0.0.' + str(version))
  runpy.run_path(path_name='website.py')