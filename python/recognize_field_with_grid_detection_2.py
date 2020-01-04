#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import griddata

from border_mapper import BorderMapper
from digit_recognizer_2 import create_recognizer
from solver import solve
from sudoku.solver import load_image, cut_out_field, find_corners, perspective_transform_contour, \
    extract_subcontour
from utils import show_image, wait_windows, scale_image, segment_length

_GRID_LINES = 10

_VIZUALIZE = True


# image = load_image("../images/big-numbers.jpg")
# image = load_image("../images/slightly_blurry.jpg")
# image = load_image("../images/sudoku.jpg")
# image = load_image("../images/sudoku-rotated.jpg")
# image = load_image("../images/sudoku-1.jpg")
# image = load_image("../images/sudoku-2.jpg")
# image = load_image("../images/sudoku-2-rotated.jpg")
image = load_image("../images/warped.jpg")
# image = load_image("tmp/001.jpg")
# image = load_image("tmp/003.jpg")
# image = load_image("tmp/005.jpg")
# image = load_image("tmp/011.jpg")
# image = load_image("tmp/100.jpg")
# image = load_image("tmp/200.jpg")
# if _VIZUALIZE:
#     show_image("orig", image)

image = scale_image(image, 640)


t = time.time()

# Extract the field, its contour and corners.
field, field_contour, _, perspective_transform_matrix = cut_out_field(image)

field_gray = cv2.cvtColor(field.image, cv2.COLOR_BGR2GRAY)
if _VIZUALIZE:
    show_image("field_gray", field_gray)

# Adjust brightness.
field_gray_closed = cv2.morphologyEx(
    field_gray, cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
field_gray_adj = np.divide(field_gray, field_gray_closed)
field_gray = cv2.normalize(field_gray_adj, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
if _VIZUALIZE:
    show_image("field_gray adj", field_gray)

# Binarize the field.
bin_field = cv2.adaptiveThreshold(
    field_gray, maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=17,
    C=11)
if _VIZUALIZE:
    show_image("bin_field", bin_field)

# Find and remove numbers. Look for them on the binary image, erase from the grayscale image.
cell_side = field.side // 9
contours, _ = cv2.findContours(bin_field, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > cell_side * 0.8 or w < 0.2 * cell_side:
        continue
    if h > cell_side * 0.9 or h < 0.2 * cell_side:
        continue
    # cv2.rectangle(field_viz, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=-1)
    # cv2.rectangle(field_viz, (x, y), (x + w, y + h), color=(0, 0, 0), thickness=-1)
    # cv2.drawContours(field_viz, [contour], 0, color=(0, 255, 0), thickness=-1)
    # cv2.drawContours(field_gray, [contour], 0, color=255, thickness=2)
    # cv2.drawContours(field_gray, [contour], 0, color=255, thickness=-1)

# field_gray = cv2.GaussianBlur(field_gray, (7, 7), 0)

if _VIZUALIZE:
    show_image("field_gray no numbers", field_gray)

# Apply the Sobel operator of 2nd degree to both directions.
grad_x = cv2.Sobel(field_gray, ddepth=cv2.CV_64F, dx=2, dy=0, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
np.clip(grad_x, a_min=0, a_max=grad_x.max(), out=grad_x)
grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

grad_y = cv2.Sobel(field_gray, ddepth=cv2.CV_64F, dx=0, dy=2, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
np.clip(grad_y, a_min=0, a_max=grad_y.max(), out=grad_y)
grad_y = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

if _VIZUALIZE:
    show_image("grad_x", grad_x)
    show_image("grad_y", grad_y)

_, grad_y_thresh = cv2.threshold(grad_y, 5, 255, cv2.THRESH_BINARY)

cells_in_sub = 3

x_from = 0
x_to = field.margin + cell_side * cells_in_sub
assert x_to > x_from

gap = 5

# sub_image = grad_y_thresh[:, s_left:s_left + cell_side * cells_in_sub]
# mask = np.zeros(grad_y_thresh.shape, dtype=np.uint8)
# for col in range(0, mask.shape[1], gap):
#     mask[:, col] = 255
# grad_y_thresh = np.bitwise_and(sub_image, mask)


import math
min_theta = 90 - 30
max_theta = 90 + 30
theta_step = 1
rho_step = 1

max_rho = image.shape[0] + (x_to - x_from)
min_rho = -max_rho

num_theta = (max_theta - min_theta) // theta_step + 1
num_rho = max_rho - min_rho + 1
hough_space = np.zeros((num_rho, num_theta), dtype=np.int32)
points = [
    [[] for _ in range(num_theta)] for _ in range(num_rho)
]
for y in range(grad_y_thresh.shape[0]):
    # TODO x should start with mapping
    for x in range(x_from, x_to, gap):
        if grad_y_thresh[y, x] == 0:
            continue
        for theta in range(min_theta, max_theta, theta_step):
            theta_rad = math.radians(theta)
            rho = int(round(
                y * math.sin(theta_rad) + x * math.cos(theta_rad)
            ))
            rho = rho // rho_step
            theta = (theta - min_theta) // theta_step
            hough_space[rho, theta] += 1
            points[rho][theta].append((x, y))


good = np.where(hough_space > (x_to - x_from) / gap / 100)  # todo no hardcoding
grad_y = cv2.cvtColor(grad_y, cv2.COLOR_GRAY2BGR)

lines = []
for i in range(good[0].shape[0]):
    rho = good[0][i]
    theta = good[1][i]
    line_points = points[rho][theta]
    line_points = sorted(line_points, key=lambda item: (item[0], item[1]))

    broken_line = False
    for j in range(len(line_points) - 1):
        x1 = line_points[j][0]
        x2 = line_points[j + 1][0]
        if abs(x1 - x2) > gap * 2.5:
            broken_line = True
            break
    if broken_line:
        continue

    [x1, y1] = line_points[0]
    [x2, y2] = line_points[-1]
    length = segment_length(x1, y1, x2, y2)
    if length / cell_side > cells_in_sub * 1.2:
        continue
    if length / cell_side < cells_in_sub * 0.6:
        continue


    rho = rho * rho_step
    theta = theta * theta_step + min_theta

    # print(rho, theta)
    # print(length / cell_side)

    lines.append((rho, theta, line_points))
    for x, y in line_points:
        cv2.circle(grad_y, (x, y), 1, (0,0,255), 1)
    # break

    # theta_rad = math.radians(theta)
    # a = np.cos(theta_rad)
    # b = np.sin(theta_rad)
    # x0 = a*rho
    # y0 = b*rho
    # x1 = int(x0 + 1000*(-b))
    # y1 = int(y0 + 1000*(a))
    # x2 = int(x0 - 1000*(-b))
    # y2 = int(y0 - 1000*(a))
    # cv2.line(sub_image,(x1,y1),(x2,y2),(0,0,255),1)

# cv2.line(grad_y, (10,10), (100, int(math.tan(math.radians(45)) * 100)), (0, 0, 255), 1)
# cv2.line(grad_y, (10,10), (100, int(math.tan(math.radians(30)) * 100)), (0, 0, 255), 1)
# cv2.line(grad_y, (10,10), (100, int(math.tan(math.radians(18)) * 100)), (0, 0, 255), 1)

lines = sorted(lines, key=lambda x: x[0])  # sort by rho

if _VIZUALIZE:
    show_image("xxx", grad_y, 700)

print((time.time() - t) * 1000)

if _VIZUALIZE:
    # show_image("unwrapped", unwrapped_viz)
    # show_image("wrapped back", wrapped_viz)
    # show_image("field_viz", field_viz, 700)
    pass

if _VIZUALIZE:
    wait_windows()
