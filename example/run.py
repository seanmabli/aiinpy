import subprocess

while True:
  print(subprocess.Popen("python3 gru-timeseries.py", shell=True, stdout=subprocess.PIPE).communicate()[0])