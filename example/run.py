import subprocess

while True:
  print(subprocess.Popen("python3 lstm-timeseries.py", shell=True, stdout=subprocess.PIPE).communicate()[0])