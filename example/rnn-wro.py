from math import sqrt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

COLORS = (
  (29, 40, 7),
  (38,  87, 61),
	(34, 122, 12),
	(238, 163, 15),
	(192,  44,  8),
	(306, 324, 110),
)

def closest_color(rgb):
  r, g, b = rgb
  color_diffs = []
  for color in COLORS:
    cr, cg, cb = color
    color_diff = sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
    color_diffs.append((color_diff, color))
  return min(color_diffs)[1]

boardimage = Image.open("data/wro/2022seniorboard.png").convert("RGB")
width, height = boardimage.size
# resize to 1/4
boardimage = boardimage.resize((width // 10, height // 10))
width, height = boardimage.size

for x in range(width):
  for y in range(height):
    pixel = boardimage.getpixel((x, y))
    boardimage.putpixel((x, y), closest_color(pixel))
    print((x * height + y) / (width * height))

boardimage.save("data/wro/2022seniorboard.png")