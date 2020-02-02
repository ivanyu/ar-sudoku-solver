# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import torchvision as tv


def draw_borders(border_prob: float):
    def inner(img: Image):
        draw = ImageDraw.Draw(img)

        max_gap = img.width // 5

        # The top.
        if random.uniform(0, 1) < border_prob:
            width = random.randint(1, 3)
            x_from = random.randint(0, max_gap)
            x_to = random.randint(img.width - max_gap, img.width)
            fill = random.randint(100, 255)
            draw.line((x_from, 0, x_to, 0), fill=fill, width=width)

        # The bottom.
        if random.uniform(0, 1) < border_prob:
            width = random.randint(1, 3)
            x_from = random.randint(0, max_gap)
            x_to = random.randint(img.width - max_gap, img.width)
            fill = random.randint(100, 255)
            draw.line((x_from, img.height, x_to, img.height), fill=fill, width=width)

        # The left.
        if random.uniform(0, 1) < border_prob:
            width = random.randint(1, 3)
            y_from = random.randint(0, max_gap)
            y_to = random.randint(img.height - max_gap, img.height)
            fill = random.randint(100, 255)
            draw.line((0, y_from, 0, y_to), fill=fill, width=width)

        # The right.
        if random.uniform(0, 1) < border_prob:
            width = random.randint(1, 3)
            y_from = random.randint(0, max_gap)
            y_to = random.randint(img.height - max_gap, img.height)
            fill = random.randint(100, 255)
            draw.line((img.width, y_from, img.width, y_to), fill=fill, width=width)

        return img
    return inner


def morphology():
    def inner(img: Image):
        img_np = np.array(img)
        img_np = cv2.morphologyEx(
            img_np,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1
        )
        return Image.fromarray(img_np)
    return inner


def scale_and_shift(img: Image):
    scale = random.uniform(0.5, 1.0)
    # scale = 1.0
    # scale = 0.5
    translation_space_x = 4
    translation_space_x = int(round(translation_space_x / scale))
    translation_x = random.randint(-translation_space_x, translation_space_x)

    original_digit_height = img.height - 2
    translation_space_y = (img.height - int(round(original_digit_height * scale))) // 2 - 1
    translation_y = random.randint(-translation_space_y, translation_space_y)

    img = tv.transforms.functional.affine(img, angle=0, translate=(translation_x, translation_y), scale=scale, shear=0, resample=Image.BICUBIC)

    return img


def gaussian_noise(mean=0, sigma=1):
    def inner(img: Image):
        img_np = np.array(img)
        rows, cols = img_np.shape

        sigma_actual = sigma
        if isinstance(sigma, tuple):
            sigma_actual = random.uniform(sigma[0], sigma[1])

        gauss = np.random.normal(mean, sigma_actual, (rows, cols))
        gauss = gauss.reshape(rows, cols)
        noisy = img_np.astype(np.float) + gauss
        noisy = cv2.normalize(noisy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return Image.fromarray(noisy)

    return inner


def gaussian_blur(radius=1):
    def inner(img: Image):
        actual_radius = radius
        if isinstance(radius, tuple):
            actual_radius = random.uniform(radius[0], radius[1])
        r = img.filter(ImageFilter.GaussianBlur(radius=actual_radius))
        return r
    return inner


def salt_and_pepper_noise():
    def inner(img: Image):
        pepper_level = random.uniform(1, 10)
        salt_level = random.uniform(253, 255)

        img_np = np.array(img)
        noise_map = np.random.uniform(0, 255, img_np.shape).astype(np.uint8)
        mask = np.bitwise_or(noise_map < pepper_level, noise_map > salt_level)
        np.putmask(img_np, mask, noise_map)
        return Image.fromarray(img_np)
    return inner


def scratches():
    def inner(img: Image):
        count = random.randint(0, 3)
        draw = ImageDraw.Draw(img)
        for _ in range(count):
            max_distance = 3
            x1 = random.randint(0, img.width)
            y1 = random.randint(0, img.height)
            x2 = random.randint(max(0, x1 - max_distance), min(x1 + max_distance, img.width))
            y2 = random.randint(max(0, y1 - max_distance), min(y1 + max_distance, img.height))
            fill = random.randint(0, 255)
            draw.line((x1, y1, x2, y2), fill=fill)
        return img
    return inner


def cross_through():
    def inner(img: Image):
        count = random.randint(1, 3)
        draw = ImageDraw.Draw(img)
        for _ in range(count):
            is_horizontal = random.randint(0, 1)
            if is_horizontal:
                x1 = 0
                y1 = random.randint(0, img.height)
                x2 = img.width
                y2 = random.randint(0, img.height)
            else:
                x1 = random.randint(0, img.width)
                y1 = 0
                x2 = random.randint(0, img.width)
                y2 = img.height
            draw.line((x1, y1, x2, y2), fill=0, width=2)

        return img
    return inner
