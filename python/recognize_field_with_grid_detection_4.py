#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import List, Tuple

import cv2
import numpy as np

from line_mapper import LineMapper
from refine_point import refine_point
from sudoku.solver import load_image, cut_out_field, find_corners, perspective_transform_contour, \
    extract_subcontour
from utils import show_image, wait_windows, scale_image_target_height, scale_image_target_width

_GRID_LINES = 10

_VIZUALIZE = True


# image = load_image("../images/big-numbers.jpg")
# image = load_image("../images/slightly_blurry.jpg")
# image = load_image("../images/sudoku.jpg")
# image = load_image("../images/sudoku-rotated.jpg")
# image = load_image("../images/sudoku-1.jpg")
# image = load_image("../images/sudoku-2.jpg")
# image = load_image("../images/sudoku-2-rotated.jpg")
# image = load_image("../images/warped.jpg")
# image = load_image("tmp/001.jpg")
# image = load_image("tmp/002.jpg")
# image = load_image("tmp/003.jpg")
# image = load_image("tmp/005.jpg")
# image = load_image("tmp/011.jpg")
# image = load_image("tmp/035.jpg")
# image = load_image("tmp/041.jpg")  # !
# image = load_image("tmp/100.jpg")
# image = load_image("tmp/200.jpg")
# image = load_image("tmp/210.jpg")
# image = load_image("tmp/220.jpg")
# image = load_image("tmp/230.jpg")
# image = load_image("tmp/240.jpg")
# image = load_image("tmp/250.jpg")
# image = load_image("tmp/262.jpg")

image = load_image("tmp/254.png")

# if _VIZUALIZE:
#     show_image("orig", image)

image = scale_image_target_height(image, 640)


def find_next_line(work_image: np.ndarray, current_line: np.ndarray, left_border_mapper: LineMapper,
                   vizualization: dict, debug: bool) -> Tuple[int, int, Tuple[float, float, float, float]]:
    x1, y1 = current_line[0]

    x_min = np.min(current_line[:, 0])
    x_max = np.max(current_line[:, 0])
    y_min = np.min(current_line[:, 1])
    y_max = np.max(current_line[:, 1])

    # x1_rel = x1 - x_min
    # y1_rel = y1 - y_min

    current_line_rel = np.copy(current_line)
    current_line_rel[:, 0] = current_line_rel[:, 0] - x_min
    current_line_rel[:, 1] = current_line_rel[:, 1] - y_min

    poly = [np.array(current_line_rel, np.int32)]
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    cv2.polylines(mask, poly, isClosed=False, color=255, thickness=1)

    max_intersect = np.count_nonzero(mask)
    intersect_threshold = max_intersect * 0.4
    print("Max intersect:", max_intersect)
    print("Intersect threshold:", intersect_threshold)

    focus_window_additional_height_half = 7

    seen_max = None
    seen_max_y_off = None
    seen_max_x_off = None
    for y_off in range(y1 + cell_side // 3, y1 + cell_side * 2):
        x_off = left_border_mapper.map_x(y_off)

        sub_img = work_image[y_off:mask.shape[0] + y_off, x_off:mask.shape[1] + x_off]
        cropped_mask = mask[:, 0:sub_img.shape[1]]
        assert sub_img.shape == cropped_mask.shape

        intersect = np.bitwise_and(sub_img, cropped_mask)
        s = np.sum(intersect) // 255

        if debug:
        #     cv2.circle(vizualization["image"], (x1_rel + x_off, y1_rel + y_off), 1, (0, 255, 0))
            print(s, intersect_threshold)

        if s >= intersect_threshold:
            if seen_max is None or s > seen_max:
                seen_max = s
                seen_max_y_off = y_off
                seen_max_x_off = x_off

        else:
            if seen_max is not None:
                print("Stopping at", y_off)
                break
    else:
        show_image("viz", vizualization["image"], 700)
        wait_windows()
        assert False

    focus_window = work_image[
                        seen_max_y_off - focus_window_additional_height_half:mask.shape[0] + seen_max_y_off + focus_window_additional_height_half,
                        seen_max_x_off:mask.shape[1] + seen_max_x_off]
    mask2 = np.zeros(focus_window.shape, dtype=np.uint8)
    mask_line = np.copy(current_line)
    mask_line[:, 0] = mask_line[:, 0] - x_min
    mask_line[:, 1] = mask_line[:, 1] - y_min + focus_window_additional_height_half
    poly = [np.array(mask_line, np.int32)]
    cv2.polylines(mask2, poly, isClosed=False, color=255, thickness=focus_window_additional_height_half)
    focus_window = np.bitwise_and(focus_window, mask2)
    focus_window_viz = cv2.cvtColor(focus_window, cv2.COLOR_GRAY2BGR)

    # new_line = []

    white_points = np.argwhere(focus_window > 0)
    xs = white_points[:, 1]
    ys = white_points[:, 0]
    coeffs = np.polyfit(xs, ys, 3)
    for x in range(0, mask.shape[1], 5):
        y = 0
        for p, k in enumerate(reversed(coeffs)):
            y += k * x ** p
        y = int(round(y))
        if debug:
            cv2.circle(focus_window_viz, (x, y), 0, color=(255, 0, 255), thickness=-1)
            cv2.circle(viz,
                       (x + seen_max_x_off, y + seen_max_y_off - focus_window_additional_height_half),
                       0, color=(255, 0, 255), thickness=-1)
        # new_line.append((
        #     x + seen_max_x_off,
        #     y + seen_max_y_off - focus_window_additional_height_half
        # ))

    if debug:
        # new_line_viz = np.copy(np.array(new_line))
        # new_line_viz[:, 0] = new_line_viz[:, 0]
        # new_line_viz[:, 1] = new_line_viz[:, 1]
        # poly = [np.array(new_line_viz, np.int32)]
        # cv2.polylines(viz, poly, isClosed=False, color=(255, 0, 255), thickness=1)
        show_image("viz", vizualization["image"], 1000)
        show_image("inspection", scale_image_target_width(focus_window_viz, 1000))
        wait_windows()
        exit()

    x_off = seen_max_x_off
    y_off = seen_max_y_off - focus_window_additional_height_half
    return x_off, y_off, tuple(coeffs)


def find_lines(work_image: np.ndarray,
               top_border: np.ndarray, bottom_border: np.ndarray, left_border: np.ndarray,
               vizualization: dict) -> List[List[Tuple[int, int]]]:
    lines = [[] for _ in range(10)]

    # Refine borders.
    for x, y in top_border:
        # y1 = refine_point(work_image, x, y, 3, 3)
        # if y1 is not None:
        #     y = y1
        lines[0].append((x, y))
    for x, y in bottom_border:
        # y1 = refine_point(work_image, x, y, 3, 3)
        # if y1 is not None:
        #     y = y1
        lines[9].append((x, y))

    current_line = top_border

    left_border_mapper = LineMapper(left_border)

    for i in range(1, 9):
        print(f"Finding line {i}")
        x_off, y_off, coeffs = find_next_line(
            work_image, current_line, left_border_mapper, vizualization, debug=False
        )
        # lines[i] = find_next_line(work_image, current_line, left_border_mapper,
        #                           vizualization, debug=False)

        border_starts_x = left_border_mapper.map_x(y_off)
        current_line = []
        lines[i] = []
        for x in range(-x_off, work_image.shape[1], 5):
            y = 0
            for p, k in enumerate(reversed(coeffs)):
                y += k * x ** p
            y = int(round(y))
            p = (x + x_off, y + y_off)
            lines[i].append(p)
            if x >= border_starts_x:
                current_line.append(p)
        current_line = np.array(current_line)
        # if _VIZUALIZE:
        #     for xy in lines[i]:
        #         cv2.circle(vizualization["image"], xy, 0, (0, 0, 255), -1)
        #     show_image("v", vizualization["image"], 1000)
        #     wait_windows()

    # # todo replace with line coeffs and proper drawing
    # # Due to noise and the border thickness, the first and last points in non-borders
    # # are likely to be outliers, replacing them with an exptrapolation.
    points_to_fit_on_extrapolation = 7
    # for line in lines[1:-1]:
    #     x = 0
    #     y = extrapolate_y(line[1:points_to_fit_on_extrapolation + 2], x)
    #     line[0] = (x, y)
    #
    #     x = last_visible_x
    #     y = extrapolate_y(line[-points_to_fit_on_extrapolation - 1:-1], x)
    #     line[-1] = (x, y)
    #
    # Extrapolate borders.
    for line_i in [0, -1]:
        line = lines[line_i]

        x = 0
        y = extrapolate_y(line[:points_to_fit_on_extrapolation + 1], x)
        line.insert(0, (x, y))

        x = work_image.shape[1] - 1
        y = extrapolate_y(line[-points_to_fit_on_extrapolation:], x)
        line.append((x, y))

    return lines


def extrapolate_y(points: List[Tuple[int, int]], x: int) -> int:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    a, b = np.polyfit(xs, ys, 1)
    y = int(round(a * x + b))
    return y


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

threshold = 5

grad_x = cv2.Sobel(field_gray, ddepth=cv2.CV_64F, dx=2, dy=0, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
np.clip(grad_x, a_min=0, a_max=grad_x.max(), out=grad_x)
grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
_, grad_x_clean = cv2.threshold(grad_x, threshold, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(grad_x_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Too short.
    if h < cell_side:
        cv2.drawContours(grad_x_clean, [contour], 0, color=0, thickness=-1)
    # Too wide and short to be a part of a line.
    elif w > cell_side * 0.3 and h < cell_side * 4:
        cv2.drawContours(grad_x_clean, [contour], 0, color=0, thickness=-1)

grad_y = cv2.Sobel(field_gray, ddepth=cv2.CV_64F, dx=0, dy=2, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
np.clip(grad_y, a_min=0, a_max=grad_y.max(), out=grad_y)
grad_y = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
_, grad_y_clean = cv2.threshold(grad_y, threshold, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(grad_y_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Too short.
    if w < cell_side * 0.9:
        cv2.drawContours(grad_y_clean, [contour], 0, color=0, thickness=-1)
    # Too high and short to be a part of a line.
    elif h > cell_side * 0.5 and w < cell_side * 4:
        cv2.drawContours(grad_y_clean, [contour], 0, color=0, thickness=-1)

if _VIZUALIZE:
    show_image("grad_x", grad_x)
    show_image("grad_x_clean", grad_x_clean)
    show_image("grad_y", grad_y)
    show_image("grad_y_clean", grad_y_clean)

viz = None
if _VIZUALIZE:
    viz = cv2.cvtColor(grad_y_clean, cv2.COLOR_GRAY2BGR)
    # viz = cv2.cvtColor(grad_x_clean, cv2.COLOR_GRAY2BGR)
    # viz = field.image
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


grad_x_rot = cv2.rotate(grad_x_clean, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
        "color_1": (0, 255, 0),
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

    if _VIZUALIZE:
        cv2.polylines(viz, poly, isClosed=False, color=(0, 255, 0), thickness=0, lineType=cv2.LINE_AA)
        # for xy in line:
        #     cv2.circle(viz, xy, 0, (0, 0, 255), -1)


if _VIZUALIZE:
    cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE, dst=viz)

horizontal_lines = find_lines(
    grad_y_clean,
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

    if _VIZUALIZE:
        cv2.polylines(viz, poly, isClosed=False, color=(0, 255, 0), thickness=0, lineType=cv2.LINE_AA)
        # for xy in line:
        #     cv2.circle(viz, xy, 0, (0, 0, 255), -1)

# TODO ? intersect one horizontal with all vertical
intersection = np.zeros(shape=(field.image.shape[0], field.image.shape[1]), dtype=np.uint8)

for i_row in range(_GRID_LINES):
    for i_col in range(_GRID_LINES):
        horizontal_mask = horizontal_line_masks[i_row]
        vertical_mask = vertical_line_masks[i_col]
        np.bitwise_and(horizontal_mask, vertical_mask, out=intersection)
        intersection_points = np.argwhere(intersection == 255)

        # No real intersection pixels - this may happen due to the staircase form of lines.
        if intersection_points.shape[0] == 0:
            horizontal_mask = np.roll(horizontal_mask, 1, axis=0)
            np.bitwise_and(horizontal_mask, vertical_mask, out=intersection)
            intersection_points = np.argwhere(intersection == 255)

        assert intersection_points.shape[0] >= 1

        x, y = np.flip(intersection_points[0])  # put x before y
        cv2.circle(viz, (x, y), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)


print((time.time() - t) * 1000)

if _VIZUALIZE:
    show_image("grad_x_viz", viz, 800)

if _VIZUALIZE:
    wait_windows()
