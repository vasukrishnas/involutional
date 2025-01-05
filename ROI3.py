from PIL import Image
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import cv2

def roi_extraction(img):
    # img = Image.open(img).convert("RGBA")
    img = Image.fromarray(np.uint8(img)).convert('RGBA')
    newsize = (224, 224)
    img = img.resize(newsize)
    bg , ma = img.resize(newsize),img.resize(newsize)
    background = Image.new("RGBA", bg.size, (0,0,0,0))
    mask = Image.new("RGBA", ma.size, 1)
    draw = ImageDraw.Draw(mask)
    draw.regular_polygon((110,110,110), 6, rotation=90, fill='green', outline=None)
    new_img = Image.composite(img, background, mask)
    arr = np.array(new_img.convert('RGB'))
    # arr = arr[:, :, ::-1]
    arr = cv2.resize(arr,(250,250))
    return arr
