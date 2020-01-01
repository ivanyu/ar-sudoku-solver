#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import torch
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


TRANSFORM_GRAYSCALE = tv.transforms.Grayscale()
TRANSFORM_TO_TENSOR = tv.transforms.ToTensor()
TRANSFORM_NORMALIZER = tv.transforms.Normalize((0.5,), (0.5,))


IMAGE_SIDE = 28

all_fonts = []
fonts_dir = "fonts"
all_fonts.append(os.path.join(fonts_dir, "Belmist.ttf"))
all_fonts.append(os.path.join(fonts_dir, "Imprimerie.ttf"))
all_fonts.append(os.path.join(fonts_dir, "FiraCode-Bold.ttf"))
all_fonts.append(os.path.join(fonts_dir, "FiraCode-Regular.ttf"))
all_fonts.append(os.path.join(fonts_dir, "Roboto-Black.ttf"))
all_fonts.append(os.path.join(fonts_dir, "Roboto-Regular.ttf"))
all_fonts.append(os.path.join(fonts_dir, "Ubuntu-R.ttf"))
all_fonts.append(os.path.join(fonts_dir, "UbuntuMono-R.ttf"))

# all_fonts.append("ariblk.ttf")
all_fonts.append("arialbd.ttf")
# all_fonts.append("arialbi.ttf")
# all_fonts.append("ariali.ttf")
all_fonts.append("arial.ttf")

all_fonts.append("courbd.ttf")
# all_fonts.append("courbi.ttf")
# all_fonts.append("couri.ttf")
all_fonts.append("cour.ttf")

all_fonts.append("timesbd.ttf")
# all_fonts.append("timesbi.ttf")
# all_fonts.append("timesi.ttf")
all_fonts.append("times.ttf")

os.makedirs("data/generated_digits", exist_ok=True)

print("Total fonts:", len(all_fonts))

all_digits = []
for font_path in all_fonts:
    if font_path.endswith("Belmist.ttf"):
        font_size = 34
    elif font_path.endswith("Imprimerie.ttf"):
        font_size = 42
    elif font_path.endswith("FiraCode-Bold.ttf"):
        font_size = 36
    elif font_path.endswith("FiraCode-Regular.ttf"):
        font_size = 36
    elif font_path.endswith("Roboto-Black.ttf"):
        font_size = 36
    elif font_path.endswith("Roboto-Regular.ttf"):
        font_size = 36
    elif font_path.endswith("Ubuntu-R.ttf"):
        font_size = 36
    elif font_path.endswith("UbuntuMono-R.ttf"):
        font_size = 42
    elif font_path.endswith("arialbd.ttf"):
        font_size = 34
    elif font_path.endswith("arial.ttf"):
        font_size = 34
    elif font_path.endswith("courbd.ttf"):
        font_size = 42
    elif font_path.endswith("cour.ttf"):
        font_size = 42
    elif font_path.endswith("timesbd.ttf"):
        font_size = 38
    elif font_path.endswith("times.ttf"):
        font_size = 38
    else:
        assert False
    # print(font_path)
    font = ImageFont.truetype(font_path, font_size)

    for digit in range(1, 10):
        os.makedirs(f"data/generated_digits/{digit}", exist_ok=True)

        img = Image.new("RGB", (IMAGE_SIDE, IMAGE_SIDE), "black")
        draw = ImageDraw.Draw(img)

        w, h = draw.textsize(str(digit), font)
        origin_x = (IMAGE_SIDE - w) // 2
        origin_y = (IMAGE_SIDE - h) // 2 - 4

        if font_path.endswith("Belmist.ttf"):
            origin_y += 3
        if font_path.endswith("arial.ttf"):
            origin_y += 2
        if font_path.endswith("arialbd.ttf"):
            origin_y += 2
        if font_path.endswith("Imprimerie.ttf"):
            origin_y -= 2

        draw.text((origin_x, origin_y), text=str(digit), color=(255, 255, 255), font=font)

        font_name = os.path.basename(font_path)
        with open(f"data/generated_digits/{digit}/{font_name}.jpg", "wb") as f:
            img.save(f)

        display_img = img
        display_img = TRANSFORM_GRAYSCALE(display_img)
        display_img = TRANSFORM_TO_TENSOR(display_img)
        display_img = TRANSFORM_NORMALIZER(display_img)
        all_digits.append(display_img)

images = all_digits


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


imshow(tv.utils.make_grid(images, nrow=9))
