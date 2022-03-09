from PIL import Image

numofelectronimages = 23
numoffalcon9images = 72
numofsoyuzimages = 59
numofspaceshuttleimages = 71

for i in range(72):
  im = Image.open('testing\\data\\rocketdataset\\960x540\\SoyuzImagesFormated\\Soyuz_' + format(i, "04") + '_Credit=Roscosmos.png')
  im.save('testing\\data\\rocketdataset\\soyuz\\' + str(i) + '.png')