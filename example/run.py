import subprocess

while True:
  print(subprocess.Popen("python3 nn-andor.py", shell=True, stdout=subprocess.PIPE).communicate()[0])