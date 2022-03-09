import subprocess

while True:
  print(subprocess.Popen("python3 cnn-mnist.py", shell=True, stdout=subprocess.PIPE).communicate()[0])