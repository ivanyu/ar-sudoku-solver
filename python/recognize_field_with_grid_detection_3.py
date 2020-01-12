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
# image = load_image("../images/sudoku-rotated.jpg")  ## 1111
# image = load_image("../images/sudoku-1.jpg")
# image = load_image("../images/sudoku-2.jpg")  ## 1111
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

# # Binarize the field.
# bin_field = cv2.adaptiveThreshold(
#     field_gray, maxValue=255,
#     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
#     thresholdType=cv2.THRESH_BINARY_INV,
#     blockSize=17,
#     C=11)
# if _VIZUALIZE:
#     show_image("bin_field", bin_field)

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


# Find the borders.

# Recalculate the contour and the corners on the perspective transformed image.
transformed_field_contour = perspective_transform_contour(field_contour, perspective_transform_matrix)
top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx = find_corners(transformed_field_contour)

# In the contour, points go counterclockwise.
# Top border: top right -> top left
# Right border: bottom right -> top right
# Bottom border: bottom left -> bottom right
# Left border: top left -> bottom left
top_border = extract_subcontour(transformed_field_contour, top_right_idx, top_left_idx)
right_border = extract_subcontour(transformed_field_contour, bottom_right_idx, top_right_idx)

# Change points order so they go from the top left corner.
top_border = np.flip(top_border, axis=0)
right_border = np.flip(right_border, axis=0)

bottom_border = extract_subcontour(transformed_field_contour, bottom_left_idx, bottom_right_idx)
left_border = extract_subcontour(transformed_field_contour, top_left_idx, bottom_left_idx)

cell_side = field.side // 9


def refine_point(work_image: np.ndarray, x: int, y: int, win_h_half: int) -> Optional[int]:
    ys_centered = np.arange(win_h_half * 2 + 1) - win_h_half
    ys_centered = ys_centered[:, np.newaxis]
    window = work_image[y - win_h_half:y + win_h_half + 1, x:x + 1]
    s = np.sum(window)
    if s > 0:
        avg = np.sum(ys_centered * window) / s
        avg = int(round(avg))
        return y + avg
    else:
        return None


def find_next_line(work_image: np.ndarray, current_line: np.ndarray, left_border_mapper: BorderMapper, horizonal_look_ahead: int,
                   vizualization: dict, debug: bool):
    x1 = current_line[0][0]
    x2 = x1 + horizonal_look_ahead
    xs = [i[0] for i in current_line]
    ys = [i[1] for i in current_line]
    a, b = np.polyfit(xs, ys, 1)
    y1 = int(round(a * x1 + b))
    y2 = int(round(a * x2 + b))

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

        # if debug:
        #     cv2.line(viz, (x1_rel + x_off, y1_rel + y_off), (x2_rel + x_off, y2_rel + y_off), (0, 255, 0), 1)
        #     print(s, intersect_threshold)

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
        show_image("viz", viz, 700)
        wait_windows()
        exit()
        assert False

    # todo optimization: don't refine if more than another threshold

    # print(seen_max, seen_max_x_off, seen_max_y_off)
    # cv2.line(viz, (x1_rel + seen_max_x_off, y1_rel + seen_max_y_off), (x2_rel + seen_max_x_off, y2_rel + seen_max_y_off), (0, 255, 0), 1)

    x = x1_rel + seen_max_x_off
    y = y1_rel + seen_max_y_off

    dx = 5
    dy = a * dx
    points_to_collect = horizonal_look_ahead // dx
    line = []
    for i in range(points_to_collect):
        x_int = int(round(x))
        y_int = int(round(y))

        y_int = refine_point(work_image, x_int, y_int, 2)
        if y_int is not None:
           line.append((x_int, y_int))

        x += dx
        y += dy
    return line


def continue_line(work_image: np.ndarray, line: List[Tuple[int, int]], vizualization: dict,
                  debug: bool) -> List[Tuple[int, int]]:
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

            # if _VIZUALIZE:
            #     if run % 2 == 0:
            #         color = vizualization["color_1"]
            #     else:
            #         color = vizualization["color_2"]
            # if _VIZUALIZE:
            #     cv2.rectangle(viz, (x_int, y_int - win_h_half), (x_int, y_int + win_h_half), color, 1)
            window = work_image[y_int - win_h_half:y_int + win_h_half + 1, x_int:x_int + 1]
            s = np.sum(window)
            if s > 0:
                avg = np.sum(ys_centered * window) / s
                avg = int(round(avg))
                # if vizualization["enabled"]:
                #     cv2.circle(vizualization["image"], (x_int, y_int + avg), 1, color, -1)
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
    return line


def find_lines(work_image: np.ndarray,
               top_border: np.ndarray, bottom_border: np.ndarray, left_border: np.ndarray,
               vizualization: dict) -> List[List[Tuple[int, int]]]:
    _, work_image_thresh = cv2.threshold(work_image, 5, 255, cv2.THRESH_BINARY)

    lines = [[] for _ in range(10)]

    # Refine borders.
    for x, y in top_border:
        y = refine_point(work_image, x, y, 2)
        if y is not None:
            lines[0].append((x, y))
            # if vizualization["enabled"]:
            #     cv2.circle(vizualization["image"], (x, y), 1, vizualization["color_0"], -1)
    for x, y in bottom_border:
        y = refine_point(work_image, x, y, 3)
        if y is not None:
            lines[9].append((x, y))
            # if vizualization["enabled"]:
            #     cv2.circle(vizualization["image"], (x, y), 1, vizualization["color_0"], -1)

    horizonal_look_ahead = cell_side * 2

    first_point_behind = np.argmax(top_border[:, 0] >= top_border[0, 0] + horizonal_look_ahead) + 1
    current_line = top_border[:first_point_behind + 1]

    left_border_mapper = BorderMapper(left_border)

    for i in range(1, 9):
        current_line = find_next_line(work_image_thresh, current_line, left_border_mapper, horizonal_look_ahead,
                                      vizualization, debug=False)
        # Due to noise and the border thickness, the first point is likely to be an outlier,
        # skipping it.
        current_line = current_line[1:]
        # if vizualization["enabled"]:
        #     for x, y in current_line[1:]:
        #         cv2.circle(vizualization["image"], (x, y), 1, vizualization["color_0"], -1)
        full_line = continue_line(work_image_thresh, current_line, vizualization, debug=False)

        # Extrapolate the beginning.
        points_to_fit = 5
        xs = [p[0] for p in full_line[:points_to_fit + 1]]
        ys = [p[1] for p in full_line[:points_to_fit + 1]]
        a, b = np.polyfit(xs, ys, 1)
        x = 0
        y = int(round(a * x + b))
        full_line.insert(0, (x, y))

        lines[i] = full_line

    return lines


viz = None
if _VIZUALIZE:
    _, grad_y_t = cv2.threshold(grad_y, 5, 255, cv2.THRESH_BINARY)
    # viz = cv2.cvtColor(grad_y_t, cv2.COLOR_GRAY2BGR)
    viz = field.image
    # viz = cv2.cvtColor(grad_y, cv2.COLOR_GRAY2BGR)
    cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE, dst=viz)

    # for x, y in right_border:
    #     cv2.circle(viz, (x, y), 1, (0, 255, 0), -1)


def rotate_border(border: np.ndarray, change_order: bool) -> np.ndarray:
    border = np.copy(border)
    # Change points order so they go from the top left corner.
    if change_order:
        border = np.flip(border, axis=0)
    # Swap x and y.
    border = np.flip(border, axis=1)
    border[:, 1] = field.image.shape[0] - border[:, 1]
    return border


grad_x_rot = cv2.rotate(grad_x, cv2.ROTATE_90_COUNTERCLOCKWISE)
right_border_rot = rotate_border(right_border, change_order=False)
left_border_rot = rotate_border(left_border, change_order=False)
top_border_rot = rotate_border(top_border, change_order=True)
vertical_lines = find_lines(
    grad_x_rot,
    right_border_rot,
    left_border_rot,
    top_border_rot,
    vizualization={
        "enabled": _VIZUALIZE,
        "image": viz,
        "color_0": (0, 255, 0),
        "color_1": (255, 0, 0),
        "color_2": (0, 0, 255),
    },
)

for line in vertical_lines:
    for i in range(len(line) - 1):
        x1, y1 = line[i]
        x2, y2 = line[i + 1]
        cv2.line(viz, (x1, y1), (x2, y2), (0, 255, 0), 1)

if _VIZUALIZE:
    cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE, dst=viz)

horizontal_lines = find_lines(
    grad_y,
    top_border,
    bottom_border,
    left_border,
    vizualization={
        "enabled": _VIZUALIZE,
        "image": viz,
        "color_0": (0, 255, 255),
        "color_1": (255, 0, 0),
        "color_2": (0, 0, 255),
    },
)

for line in horizontal_lines:
    for i in range(len(line) - 1):
        x1, y1 = line[i]
        x2, y2 = line[i + 1]
        cv2.line(viz, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Extract borders
# Refine borders
# For each orientation:
#  - take top border start
#  - find line beginnings
#  - continue lines
#  - extrapolate skipping the first and the last lines

print((time.time() - t) * 1000)

if _VIZUALIZE:
    show_image("grad_x_viz", viz, 700)

if _VIZUALIZE:
    wait_windows()
