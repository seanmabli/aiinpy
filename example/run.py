import subprocess

while True:
  print(subprocess.Popen("python3 rnn-posnegcon.py", shell=True, stdout=subprocess.PIPE).communicate()[0])