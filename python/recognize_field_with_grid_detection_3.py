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


def continue_line(work_image: np.ndarray, line: List[Tuple[int, int]],
                  vizualization: dict, debug: bool) -> List[Tuple[int, int]]:
    line = list(line)
    for run in range(1000):  # effectively infinite
        tail = np.array(line[-10:])
        a, b = np.polyfit(tail[:, 0], tail[:, 1], 1)
        x, y = tail[-1]
        dx = 5
        dy = a * dx

        if x >= field.margin + field.side:
            break

        new_points = continue_line_run(work_image, x, y, dx, dy, vizualization, debug)
        if len(new_points) == 0:
            break
        line += new_points
    return line


def continue_line_run(work_image: np.ndarray, x: int, y: int, dx: float, dy: float,
                      vizualization: dict, debug: bool) -> List[Tuple[int, int]]:
    new_points = []
    skipped_steps = 0
    for _ in range(40):
        x += dx

        if x >= field.margin + field.side:
            break

        y += dy
        x_int = int(round(x))
        y_int = int(round(y))

        # if _VIZUALIZE:
        #     if run % 2 == 0:
        #         color = vizualization["color_1"]
        #     else:
        #         color = vizualization["color_2"]
        # if debug and _VIZUALIZE:
        #     cv2.line(viz, (x_int, y_int - 1), (x_int, y_int + 1), (255, 0, 255), 1)
        #     # cv2.circle(viz, (x_int, y_int), 1, (255, 0, 255), -1)

        y_refined = refine_point(work_image, x_int, y_int, 1)
        if y_refined is not None:
            new_points.append((x_int, y_refined))
            # We started to diverge, let's reset.
            if y_refined != y_int:
                break
        else:
            skipped_steps += 1
            if skipped_steps >= 2:
                break
            else:
                continue
    return new_points


def find_lines(work_image: np.ndarray,
               top_border: np.ndarray, bottom_border: np.ndarray, left_border: np.ndarray,
               vizualization: dict) -> List[List[Tuple[int, int]]]:
    points_to_fit_on_extrapolation = 5

    _, work_image_thresh = cv2.threshold(work_image, 5, 255, cv2.THRESH_BINARY)

    lines = [[] for _ in range(10)]

    # Refine borders.
    for x, y in top_border:
        y = refine_point(work_image, x, y, 2)
        if y is not None:
            lines[0].append((x, y))
    for x, y in bottom_border:
        y = refine_point(work_image, x, y, 3)
        assert y is not None
        if y is not None:
            lines[9].append((x, y))

    horizonal_look_ahead = cell_side * 2

    first_point_behind = np.argmax(top_border[:, 0] >= top_border[0, 0] + horizonal_look_ahead) + 1
    current_line = top_border[:first_point_behind + 1]

    left_border_mapper = BorderMapper(left_border)

    for i in range(1, 9):
        current_line = find_next_line(work_image_thresh, current_line, left_border_mapper, horizonal_look_ahead,
                                      vizualization, debug=False)
        # if vizualization["enabled"]:
        #     for x, y in current_line[1:]:
        #         cv2.circle(vizualization["image"], (x, y), 1, vizualization["color_0"], -1)

        # Due to noise and the border thickness, the first point is likely to be an outlier,
        # replacing it with an exptrapolation.
        xs = [p[0] for p in current_line[1:points_to_fit_on_extrapolation + 2]]
        ys = [p[1] for p in current_line[1:points_to_fit_on_extrapolation + 2]]
        a, b = np.polyfit(xs, ys, 1)
        x = 0
        y = int(round(a * x + b))
        current_line[0] = (x, y)

        full_line = continue_line(work_image_thresh, current_line, vizualization, debug=i == 4)

        # Due to noise and the border thickness, the last point is likely to be an outlier,
        # replacing it with an exptrapolation.
        xs = [p[0] for p in full_line[-points_to_fit_on_extrapolation - 1:-1]]
        ys = [p[1] for p in full_line[-points_to_fit_on_extrapolation - 1:-1]]
        a, b = np.polyfit(xs, ys, 1)
        x = field.image.shape[1]
        y = int(round(a * x + b))
        full_line[-1] = (x, y)

        lines[i] = full_line

    # Extrapolate borders.
    for line_i in [0, -1]:
        line = lines[line_i]

        x = 0
        y = extrapolate_y(line[:points_to_fit_on_extrapolation + 1], x)
        line.insert(0, (x, y))

        x = work_image.shape[1]
        y = extrapolate_y(line[-points_to_fit_on_extrapolation:], x)
        line.append((x, y))

    return lines


def extrapolate_y(points: List[Tuple[int, int]], x: int) -> int:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    a, b = np.polyfit(xs, ys, 1)
    y = int(round(a * x + b))
    return y


viz = None
if _VIZUALIZE:
    # _, grad_y_t = cv2.threshold(grad_y, 5, 255, cv2.THRESH_BINARY)
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

vertical_line_masks = []
for i, line in enumerate(vertical_lines):
    poly = [np.array(line, np.int32)]
    mask = np.zeros(shape=(field.image.shape[0], field.image.shape[1]), dtype=np.uint8)
    cv2.polylines(mask, poly, isClosed=False, color=255, thickness=1)
    cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE, dst=mask)
    # Prepending is a part of rotation.
    vertical_line_masks.insert(0, mask)

    # if _VIZUALIZE:
    #     # for xy in line:
    #     #     cv2.circle(viz, xy, 1, (0, 0, 255), -1)
    #     cv2.polylines(viz, poly, isClosed=False, color=(0, 255, 0), thickness=1)

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

horizontal_line_masks = []
for i, line in enumerate(horizontal_lines):
    poly = [np.array(line, np.int32)]
    mask = np.zeros(shape=(field.image.shape[0], field.image.shape[1]), dtype=np.uint8)
    cv2.polylines(mask, poly, isClosed=False, color=255, thickness=1)
    horizontal_line_masks.append(mask)

    # if _VIZUALIZE:
    #     # for xy in line:
    #     #     cv2.circle(viz, xy, 1, (0, 0, 255), -1)
    #     cv2.polylines(viz, poly, isClosed=False, color=(0, 255, 0), thickness=1)

# TODO ? intersect one horizontal with all vertical
intersection = np.zeros(shape=(field.image.shape[0], field.image.shape[1]), dtype=np.uint8)

for i_row in range(_GRID_LINES):
    for i_col in range(_GRID_LINES):
        np.bitwise_and(horizontal_line_masks[i_row], vertical_line_masks[i_col], out=intersection)
        intersection_points = np.argwhere(intersection == 255)

        # There should not be more than several intersection points:
        # one is the default, 2 might be due to the overlapping of steps in lines.
        assert 1 <= intersection_points.shape[0] <= 2

        x, y = np.flip(intersection_points[0])  # put x before y
        cv2.circle(viz, (x, y), 1, (0, 0, 255), -1)


print((time.time() - t) * 1000)

if _VIZUALIZE:
    show_image("grad_x_viz", viz, 700)

if _VIZUALIZE:
    wait_windows()
