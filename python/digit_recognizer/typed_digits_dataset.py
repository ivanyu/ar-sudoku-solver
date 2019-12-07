# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision as tv


class TypedDigits(tv.datasets.vision.VisionDataset):
    _IMAGE_SIDE = 28

    def __init__(self, transform):
        super(TypedDigits, self).__init__(self, transform=transform)

        self.data = []
        self.targets = []

        all_fonts = []
        fonts_dir = "fonts"
        for f in os.listdir(fonts_dir):
            if f.endswith(".ttf"):
                all_fonts.append(os.path.join(fonts_dir, f))

        all_fonts.append("ariblk.ttf")
        all_fonts.append("arialbd.ttf")
        all_fonts.append("arialbi.ttf")
        all_fonts.append("ariali.ttf")
        all_fonts.append("arial.ttf")

        all_fonts.append("courbd.ttf")
        all_fonts.append("courbi.ttf")
        all_fonts.append("couri.ttf")
        all_fonts.append("cour.ttf")

        all_fonts.append("timesbd.ttf")
        all_fonts.append("timesbi.ttf")
        all_fonts.append("timesi.ttf")
        all_fonts.append("times.ttf")
        for font_name in all_fonts:
            for digit in range(1, 10):
                img = Image.new("RGB", (self._IMAGE_SIDE, self._IMAGE_SIDE), "black")
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype(font_name, 25)
                draw.text((self._IMAGE_SIDE // 4, 0), text=str(digit), align="center", color=(255, 255, 255), font=font)
                img = img.convert("L")
                img = img.point(lambda p: p > 70 and 255)
                self.data.append(img)
                self.targets.append(digit - 1)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    transform_augment = tv.transforms.Compose([
        tv.transforms.RandomAffine(degrees=(-6, 3), translate=(0.05, 0.05), shear=10),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = TypedDigits(transform=transform_augment)
    loader = torch.utils.data.DataLoader(dataset, batch_size=300,
                                         shuffle=False, num_workers=0)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    imshow(tv.utils.make_grid(images))
