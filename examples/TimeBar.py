from alive_progress import alive_bar
with alive_bar(1000) as bar:
  for i in range(1000):
    bar()