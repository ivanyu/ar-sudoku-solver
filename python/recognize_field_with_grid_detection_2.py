#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from border_mapper import BorderMapper
# from digit_recognizer_2 import create_recognizer
from solver import solve
from sudoku.solver import load_image, cut_out_field, find_corners, perspective_transform_contour, \
    extract_subcontour
from utils import show_image, wait_windows, scale_image, segment_length, get_line_coeffs

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


# Find the left and the top borders.

# Recalculate the contour and the corners on the perspective transformed image.
transformed_field_contour = perspective_transform_contour(field_contour, perspective_transform_matrix)
top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx = find_corners(transformed_field_contour)

# In the contour, points go counterclockwise.
# Top border: top right -> top left
# Right border: bottom right -> top right
# Bottom border: bottom left -> bottom right
# Left border: top left -> bottom left
top_border = extract_subcontour(transformed_field_contour, top_right_idx, top_left_idx)
# Change points order so they go from the top left corner.
top_border = np.flip(top_border, axis=0)
# # Swap x and y.
# top_border = np.flip(top_border, axis=1)
# right_border = extract_border(bottom_right_idx, top_right_idx)
bottom_border = extract_subcontour(transformed_field_contour, bottom_left_idx, bottom_right_idx)
left_border = extract_subcontour(transformed_field_contour, top_left_idx, bottom_left_idx)

cell_side = field.side // 9


def find_lines_1(work_image: np.ndarray, viz: np.ndarray, border: np.ndarray, border_mapper: BorderMapper):
    gap = 5
    max_x_dist_between_points = gap * 3
    # y_center = 0
    y_center = border[0][1]
    y_center_max = border[-1][1]
    while y_center <= y_center_max:
        x_center = border_mapper.map_x(y_center)
        b = cell_side * 2

        lines = []
        for theta in range(0, 30 + 1):
            theta_tan = math.tan(math.radians(theta))

            line_top = []
            line_bottom = []
            top_wasted = False
            bottom_wasted = False
            for x_rel in range(0, b, gap):
                if top_wasted and bottom_wasted:
                    break

                x = x_rel + x_center

                if not top_wasted:
                    if (line_top and x - line_top[-1][0] < max_x_dist_between_points
                            or not line_top and x_rel < max_x_dist_between_points):
                        y_top = y_center - int(round(theta_tan * x_rel))
                        if y_top >= 0 and work_image[y_top, x] > 0:
                            line_top.append((x, y_top))
                    else:
                        top_wasted = True

                if not bottom_wasted:
                    if (line_bottom and x - line_bottom[-1][0] < max_x_dist_between_points
                            or not line_bottom and x_rel < max_x_dist_between_points):
                        y_bottom = y_center + int(round(theta_tan * x_rel))
                        if y_bottom < work_image.shape[0] and work_image[y_bottom, x] > 0:
                            line_bottom.append((x, y_bottom))
                    else:
                        bottom_wasted = True

            if not top_wasted and line_top:
                lines.append(line_top)
            # Prevent double adding.
            if not bottom_wasted and theta != 0 and line_bottom:
                lines.append(line_bottom)

        if not lines:
            y_center += 1
            continue

        max_line = max(lines, key=len)
        if len(max_line) > int(b // gap * 0.8):
            if _VIZUALIZE:
                for x, y in max_line:
                    cv2.circle(viz, (x, y), 1, (0, 0, 255), 1)
            y_center += cell_side // 2
        else:
            y_center += 1


def find_lines_2(work_image: np.ndarray, viz: np.ndarray, border: np.ndarray, border_mapper: BorderMapper):
    border_start_x, border_start_y = border[0]
    border_end_x, border_end_y = border[-1]

    min_theta = 90 - 30
    max_theta = 90 + 30

    max_rho = work_image.shape[0] + work_image.shape[1]
    min_rho = -max_rho

    theta_step = 1
    rho_step = 1

    cells_in_sub = 3
    gap = 5

    num_theta = (max_theta - min_theta) // theta_step + 1
    num_rho = (max_rho - min_rho) // theta_step + 1
    hough_space = np.zeros((num_rho, num_theta), dtype=np.int32)
    hough_space2 = [
        [0 for _ in range(num_theta)] for _ in range(num_rho)
    ]
    points = [
        [[] for _ in range(num_theta)] for _ in range(num_rho)
    ]

    num_theta_sin = [None] * (max_theta - min_theta + 1)
    num_theta_cos = [None] * (max_theta - min_theta + 1)

    thetas = np.arange(min_theta, max_theta + 1, theta_step)
    theta_rads = np.radians(thetas)
    theta_sin = np.sin(theta_rads)
    theta_cos = np.cos(theta_rads)
    theta_off = (thetas - min_theta) // theta_step

    y_from = min(border_start_y, border_end_y)
    y_to = max(border_start_y, border_end_y)
    for y in range(y_from, y_to + 1):
        x_from = border_mapper.map_x(y)
        x_to = int(x_from + cell_side * cells_in_sub)

        y_sin = theta_sin * y

        for x in range(x_from, x_to + 1, gap):
            if work_image[y, x] == 0:
                continue

            rhos = theta_cos * x
            np.add(rhos, y_sin, out=rhos)
            rhos = np.int32(rhos)
            hough_space[rhos, theta_off] += 1

            # for theta in range(min_theta, max_theta, theta_step):
            #     theta_rad = math.radians(theta)
            #     rho = int(round(
            #         y * math.sin(theta_rad) + x * math.cos(theta_rad)
            #     ))
            #     rho = rho // rho_step
            #     theta_off = (theta - min_theta) // theta_step
            #     # hough_space[rho, theta_off] += 1
            #     hough_space2[rho][theta_off] += 1
            #     points[rho][theta_off].append((x, y))

    good = np.where(hough_space > (cell_side * cells_in_sub) / (gap * 1.1))  # todo no hardcoding
    for i in range(good[0].shape[0]):
        rho = good[0][i]
        theta = good[1][i]
        theta = theta * theta_step + min_theta
        theta_rad = math.radians(theta)
        a = np.cos(theta_rad)
        b = np.sin(theta_rad)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(viz, (x1,y1), (x2,y2), (0,0,255), 1)
    pass

_, grad_y_thresh = cv2.threshold(grad_y, 5, 255, cv2.THRESH_BINARY)
top_border_mapper = BorderMapper(top_border)

_, grad_x_thresh = cv2.threshold(grad_x, 5, 255, cv2.THRESH_BINARY)
grad_x_thresh = cv2.rotate(grad_x_thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
left_border_mapper = BorderMapper(left_border)

work_image_orig = grad_y
work_image = grad_y_thresh
# work_image = grad_x_thresh
border = left_border
# border = top_border
border_mapper = left_border_mapper
# border_mapper = top_border_mapper

viz = cv2.cvtColor(work_image, cv2.COLOR_GRAY2BGR)

# find_lines_1(work_image, viz, border, border_mapper)
# for _ in range(1000):
#     find_lines_1(work_image, viz, border, border_mapper)
    # find_lines_2(work_image, viz, border, border_mapper)
# find_lines_2(work_image, viz, border, border_mapper)

def find_next_line(current_line: np.ndarray, horizonal_look_ahead: int):
    x1 = top_border[0, 0]
    x2 = x1 + horizonal_look_ahead
    a, b = np.polyfit(current_line[:, 0], current_line[:, 1], 1)
    y1 = int(round(a * x1 + b))
    y2 = int(round(a * x2 + b))

    # print(math.degrees(math.atan(a)))

    def _relative_coords(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        assert x1 < x2
        x2 -= x1
        x1 = 0
        y_min = min(y1, y2)
        y1 -= y_min
        y2 -= y_min
        return x1, y1, x2, y2


    x1_rel, y1_rel, x2_rel, y2_rel = _relative_coords(x1, y1, x2, y2)


    def create_mask(x1_rel: int, y1_rel: int, x2_rel: int, y2_rel: int) -> Tuple[np.ndarray, int, int]:
        assert x1_rel < x2_rel
        mask = np.zeros((abs(y1_rel - y2_rel) + 1, x2_rel - x1_rel + 1), dtype=np.uint8)
        cv2.line(mask, (x1_rel, y1_rel), (x2_rel, y2_rel), 255, 1)
        return mask


    mask = create_mask(x1_rel, y1_rel, x2_rel, y2_rel)
    max_intersect = np.count_nonzero(mask)
    intersect_threshold = max_intersect * 0.7

    seen_max = None
    seen_max_y_off = None
    seen_max_x_off = None
    for y_off in range(y1 + cell_side // 3, y1 + cell_side * 2):
        x_off = left_border_mapper.map_x(y_off)

        sub_img = work_image[y_off:mask.shape[0] + y_off, x_off:mask.shape[1] + x_off]
        assert sub_img.shape == mask.shape

        intersect = np.bitwise_and(sub_img, mask)
        s = np.sum(intersect) // 255

        if s >= intersect_threshold:
            if seen_max is None or s > seen_max:
                seen_max = s
                seen_max_y_off = y_off
                seen_max_x_off = x_off

            # print(y_off, s)
            # cv2.line(viz, (x1_rel + x_off, y1_rel + y_off), (x2_rel + x_off, y2_rel + y_off), (0, 255, 0), 1)
        else:
            if seen_max is not None:
                # print("Stopping at", y_off)
                break
    else:
        assert False

    # todo optimization: don't refine if more than another threshold

    # print(seen_max, seen_max_x_off, seen_max_y_off)
    # cv2.line(viz, (x1_rel + seen_max_x_off, y1_rel + seen_max_y_off), (x2_rel + seen_max_x_off, y2_rel + seen_max_y_off), (0, 255, 0), 1)

    win_h_half = 2

    ys_centered = np.arange(win_h_half * 2 + 1) - win_h_half
    ys_centered = ys_centered[:, np.newaxis]

    x = x1_rel + seen_max_x_off
    y = y1_rel + seen_max_y_off

    dx = 5
    dy = a * dx
    points_to_collect = horizonal_look_ahead // dx
    line = []
    for i in range(points_to_collect):
        x_int = int(round(x))
        y_int = int(round(y))

        y_int = refine_point(x_int, y_int, 2,)
        if y_int is not None:
           line.append((x_int, y_int))

        # # cv2.rectangle(viz, (x_int, y_int - win_h_half), (x_int, y_int + win_h_half), (0, 255, 0), 1)
        # window = work_image_orig[y_int - win_h_half:y_int + win_h_half + 1, x_int:x_int + 1]
        # s = np.sum(window)
        # if s > 0:
        #     avg = np.sum(ys_centered * window) / s
        #     avg = int(round(avg))
        #     # print(avg)
        #     line.append((x_int, y_int + avg))

        x += dx
        y += dy
    return np.array(line)


def continue_line(line: List[np.ndarray]):
    win_h_half = 1

    ys_centered = np.arange(win_h_half * 2 + 1) - win_h_half
    ys_centered = ys_centered[:, np.newaxis]

    line = list(line)
    last_len = len(line)

    for run in range(1000):  # effectively infinite
        tail = np.array(line[-5:])
        a, b = np.polyfit(tail[:, 0], tail[:, 1], 1)
        x, y = tail[-1]
        dx = 5
        dy = a * dx

        if x >= field.margin + field.side:
            break

        skipped_steps = 0
        for _ in range(40):
            x += dx

            if x >= field.margin + field.side:
                break

            y += dy
            x_int = int(round(x))
            y_int = int(round(y))
            # cv2.circle(viz, (x_int, y_int), 2, (0, 0, 255), -1)
            # print(work_image[y_int, x_int])

            if run % 2 == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            # if _VIZUALIZE:
            #     cv2.rectangle(viz, (x_int, y_int - win_h_half), (x_int, y_int + win_h_half), color, 1)
            window = work_image_orig[y_int - win_h_half:y_int + win_h_half + 1, x_int:x_int + 1]
            s = np.sum(window)
            if s > 0:
                avg = np.sum(ys_centered * window) / s
                avg = int(round(avg))
                if _VIZUALIZE:
                    cv2.circle(viz, (x_int, y_int + avg), 2, color, -1)
                line.append((x_int, y_int + avg))
                if avg != 0:
                    break
            else:
                skipped_steps += 1
                if skipped_steps >= 2:
                    break
                else:
                    continue

        if len(line) == last_len:
            break
        last_len = len(line)


def refine_point(x: int, y: int, win_h_half: int) -> Optional[int]:
    ys_centered = np.arange(win_h_half * 2 + 1) - win_h_half
    ys_centered = ys_centered[:, np.newaxis]
    window = work_image_orig[y - win_h_half:y + win_h_half + 1, x:x + 1]
    s = np.sum(window)
    if s > 0:
        avg = np.sum(ys_centered * window) / s
        avg = int(round(avg))
        return y + avg
    else:
        return None


horizonal_look_ahead = cell_side * 2

first_point_behind = np.argmax(top_border[:, 0] >= top_border[0, 0] + horizonal_look_ahead) + 1
current_line = top_border[:first_point_behind + 1]

if _VIZUALIZE:
    for x, y in top_border:
        y1 = refine_point(x, y, 2)
        if y1 is not None:
            y = y1
            cv2.circle(viz, (x, y), 1, (0, 255, 0), -1)

for i in range(8):
    # Due to noise and the border thickness, the first point is likely to be an outlier,
    # skipping it.
    current_line = current_line[1:, :]
    current_line = find_next_line(current_line, horizonal_look_ahead)
    if _VIZUALIZE:
        for x, y in current_line[1:]:
            cv2.circle(viz, (x, y), 1, (0, 255, 0), -1)
    continue_line(current_line)

if _VIZUALIZE:
    for x, y in bottom_border:
        y1 = refine_point(x, y, 2)
        if y1 is not None:
            y = y1
            cv2.circle(viz, (x, y), 1, (0, 255, 0), -1)

print()
print((time.time() - t) * 1000)

if _VIZUALIZE:
    show_image("grad_x_viz", viz, 700)
if _VIZUALIZE:
    wait_windows()
exit()

cells_in_sub = 3

x_from = 0
x_to = field.margin + cell_side * cells_in_sub
assert x_to > x_from

gap = 5


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
for y_center in range(grad_y_thresh.shape[0]):
    # TODO x should start with mapping
    for x_center in range(x_from, x_to, gap):
        if grad_y_thresh[y_center, x_center] == 0:
            continue
        for theta in range(min_theta, max_theta, theta_step):
            theta_rad = math.radians(theta)
            rho = int(round(
                y_center * math.sin(theta_rad) + x_center * math.cos(theta_rad)
            ))
            rho = rho // rho_step
            theta = (theta - min_theta) // theta_step
            hough_space[rho, theta] += 1
            points[rho][theta].append((x_center, y_center))


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

    [x1, y_top] = line_points[0]
    [x2, y2] = line_points[-1]
    length = segment_length(x1, y_top, x2, y2)
    if length / cell_side > cells_in_sub * 1.2:
        continue
    if length / cell_side < cells_in_sub * 0.6:
        continue


    rho = rho * rho_step
    theta = theta * theta_step + min_theta

    # print(rho, theta)
    # print(length / cell_side)

    lines.append((rho, theta, line_points))
    for x_center, y_center in line_points:
        cv2.circle(grad_y, (x_center, y_center), 1, (0, 0, 255), 1)
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
