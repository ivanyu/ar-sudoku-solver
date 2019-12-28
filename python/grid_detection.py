#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import griddata

from border_mapper import BorderMapper
from digit_recognizer import recognize_digits
from solver import solve
from sudoku.solver import load_image, cut_out_field, find_field_corners, perspective_transform_contour, \
    extract_subcontour
from utils import show_image, wait_windows, scale_image

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
# if _VIZUALIZE:
#     show_image("orig", image)

image = scale_image(image, 640)


t = time.time()

# Extract the field, its contour and corners.
field, field_contour, field_corners, perspective_transform_matrix = cut_out_field(image)

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

# field_viz = cv2.cvtColor(field_gray, cv2.COLOR_GRAY2BGR)

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
    cv2.drawContours(field_gray, [contour], 0, color=255, thickness=2)
    cv2.drawContours(field_gray, [contour], 0, color=255, thickness=-1)

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


# TODO adaptive window
# TODO fast-and-simple first, fallback on more robust after?

# Outline grid lines.
# The horizontal one, on the left border. The vertical one, on the top border.
def get_average_y_over_window(image, win_x: int, win_y: int, win_w: int, win_h: int) -> Optional[int]:
    window = image[win_y:win_y + win_h, win_x:win_x + win_w]
    xs, ys = np.meshgrid(np.arange(window.shape[1]), np.arange(window.shape[0]))
    xs_dist_weight = np.exp2(xs)
    pure_white = np.sum(window.shape[0] * window.shape[1] * 255 * xs_dist_weight)
    weighted_window = window * xs_dist_weight
    current_white = np.sum(weighted_window)
    frac = current_white / pure_white
    if frac < 0.0001:
        return None
    avg = np.sum(ys * weighted_window) / current_white
    return int(round(avg))


def detect_line(image, start_x, start_y, win_h, win_w, right_limit) -> Optional[List[Tuple[int, int]]]:
    """
    Detects a horizontal white line.
    """

    result: List[Tuple[int, int]] = []

    win_x = start_x
    win_y = start_y - win_h // 2

    lost_line = False
    while win_x < right_limit:
        current_win_w = win_w

        if win_x + current_win_w >= right_limit:
            break
        # Commented-out note:
        # Alternatively, try to use a smaller window.
        # However, it seems just stopping works better with line extrapolation later.
        # if win_x + current_win_w > right_limit:
        #     current_win_w = right_limit - win_x
        #     if current_win_w < 3:
        #         break

        avg = get_average_y_over_window(image, win_x, win_y, current_win_w, win_h)
        if avg is None:
            print("EXPAND")
            win_y -= win_h // 2
            avg = get_average_y_over_window(image, win_x, win_y, current_win_w, win_h * 2)

        if avg is None:
            print("LOST")
            lost_line = True

        win_x = win_x + current_win_w
        win_y = win_y + (avg - win_h // 2)
        result.append((win_x, win_y + win_h // 2))

    if not lost_line:
        return result
    else:
        return None


def detect_lines(work_image, border_mapper, right_limit) -> List[List[Tuple[int, int]]]:
    work_image_blur = cv2.GaussianBlur(work_image, (1, 25), 0)

    offset = cell_side // 6
    # offset = 0
    step = 1
    win_w = cell_side // 4

    detected_windows = []
    for y in range(field.margin, field.margin + field.side + 1, step):
        x = border_mapper.map_x(y) + offset
        w = work_image_blur[y - 3:y + 3, x:x + win_w]

        pure_white = w.shape[0] * w.shape[1] * 255
        current_white = np.sum(w)
        frac = current_white / pure_white
        if frac > 0.01:
            detected_windows.append((x, y, frac))

    assert len(detected_windows) >= _GRID_LINES

    cluster_starts = [0]
    for i in range(1, len(detected_windows)):
        if detected_windows[i][1] - detected_windows[i - 1][1] > cell_side // 6:
            # print("---")
            cluster_starts.append(i)
        # print(detected_windows[i])

    # print(cluster_starts)
    assert len(cluster_starts) == _GRID_LINES

    win_h = 5
    win_w = cell_side // 4
    result = []
    for i in range(len(cluster_starts)):
        if i < len(cluster_starts) - 1:
            windows = detected_windows[cluster_starts[i]:cluster_starts[i + 1]]
        else:
            windows = detected_windows[cluster_starts[i]:]

        x, y, _ = max(windows, key=lambda w: w[2])
        # if _VIZUALIZE:
        #     cv2.rectangle(viz, (x, y - 3), (x + win_w, y + 3), color=(0, 255, 0), thickness=1)
        line = detect_line(work_image, x, y, win_h, win_w, right_limit)

        # Extrapolate the beginning and the end.
        points_to_fit = 3
        xs = [p[0] for p in line[:points_to_fit + 1]]
        ys = [p[1] for p in line[:points_to_fit + 1]]
        a, b = np.polyfit(xs, ys, 1)
        x = 0
        y = int(round(a * x + b))
        line.insert(0, (x, y))

        xs = [p[0] for p in line[-points_to_fit:]]
        ys = [p[1] for p in line[-points_to_fit:]]
        a, b = np.polyfit(xs, ys, 1)
        x = field_viz.shape[1]
        y = int(round(a * x + b))
        line.append((x, y))

        result.append(line)
        # break
    return result


# Find the left and the top borders.

# Recalculate the contour and the corners on the perspective transformed image.
transformed_field_contour = perspective_transform_contour(field_contour, perspective_transform_matrix)
top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx = find_field_corners(transformed_field_contour)

# In the contour, points go counterclockwise.
# Top border: top right -> top left
# Right border: bottom right -> top right
# Bottom border: bottom left -> bottom right
# Left border: top left -> bottom left
top_border = extract_subcontour(transformed_field_contour, top_right_idx, top_left_idx)
# Change points order so they go from the top left corner.
top_border = np.flip(top_border, axis=0)
# Swap x and y.
top_border = np.flip(top_border, axis=1)
# right_border = extract_border(bottom_right_idx, top_right_idx)
# bottom_border = extract_border(bottom_left_idx, bottom_right_idx)
left_border = extract_subcontour(transformed_field_contour, top_left_idx, bottom_left_idx)

field_viz = cv2.cvtColor(field_gray, cv2.COLOR_GRAY2BGR)
cv2.rotate(field_viz, cv2.ROTATE_90_COUNTERCLOCKWISE, dst=field_viz)

top_border_mapper = BorderMapper(top_border)
vertical_lines = detect_lines(
    cv2.rotate(grad_x, cv2.ROTATE_90_COUNTERCLOCKWISE),
    top_border_mapper,
    field.margin + field.side
)
assert len(vertical_lines) == _GRID_LINES

vertical_lines_masks = np.zeros(shape=(_GRID_LINES, field.image.shape[0], field.image.shape[1]), dtype=np.uint8)
for i, line in enumerate(vertical_lines):
    poly = [np.array(line, np.int32)]
    # if _VIZUALIZE:
    #     for x, y in line:
    #         cv2.circle(field_viz, (x, y), 0, (0, 0, 255), 2)
    #     cv2.polylines(field_viz, poly, isClosed=False, color=(0, 0, 255), thickness=1)

    # Invert the index: the first in the rotated image is the last by the normal order.
    inv_i = _GRID_LINES - i - 1
    cv2.polylines(vertical_lines_masks[inv_i], poly, isClosed=False, color=255, thickness=1)

# TODO rotate lines before drawing
vertical_lines_masks = np.rot90(vertical_lines_masks, k=-1, axes=(1, 2))

if _VIZUALIZE:
    cv2.rotate(field_viz, cv2.ROTATE_90_CLOCKWISE, dst=field_viz)

left_border_mapper = BorderMapper(left_border)
horizontal_lines = detect_lines(grad_y, left_border_mapper, field.margin + field.side)
assert len(horizontal_lines) == _GRID_LINES

horizontal_lines_masks = np.zeros(shape=(_GRID_LINES, field.image.shape[0], field.image.shape[1]), dtype=np.uint8)
for i, line in enumerate(horizontal_lines):
    poly = [np.array(line, np.int32)]
    # if _VIZUALIZE:
    #     for x, y in line:
    #         cv2.circle(field_viz, (x, y), 0, (255, 255, 0), 2)
    #     cv2.polylines(field_viz, poly, isClosed=False, color=(255, 255, 0), thickness=1)
    cv2.polylines(horizontal_lines_masks[i], poly, isClosed=False, color=255, thickness=1)

# # TODO ? intersect one horizontal with all vertical
intersection = np.zeros(shape=(field.image.shape[0], field.image.shape[1]), dtype=np.uint8)
grid_points = np.zeros(shape=(_GRID_LINES, _GRID_LINES, 2), dtype=np.uint32)

src_points = []
dst_points = []

for i_row in range(_GRID_LINES):
    for i_col in range(_GRID_LINES):
        np.bitwise_and(horizontal_lines_masks[i_row], vertical_lines_masks[i_col], out=intersection)
        intersection_points = np.argwhere(intersection == 255)

        # There should not be more than several intersection points:
        # one is the default, 2 might be due to the overlapping of steps in lines.
        assert 1 <= intersection_points.shape[0] <= 2
        grid_points[i_row, i_col] = np.flip(intersection_points[0])  # put x before y

        # src_points.append(np.flip(intersection_points[0]))
        src_points.append(intersection_points[0])
        dst_points.append((i_row * cell_side, i_col * cell_side))

        if _VIZUALIZE:
            y, x = intersection_points[0]
            cv2.circle(field_viz, (x, y), radius=1, color=(100, 100, 255), thickness=-1)
            # cv2.putText(field_viz,
            #             text=f"{i_row},{i_col}",
            #             org=(x, y),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.3,
            #             color=(0, 0, 255))

# Unwrap the grid.
grid_x, grid_y = np.mgrid[0:field.side, 0:field.side]
grid_z = griddata(np.array(dst_points), np.array(src_points), (grid_x, grid_y), method='cubic')
map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(field.side, field.side)
map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(field.side, field.side)
map_x_32 = map_x.astype('float32')
map_y_32 = map_y.astype('float32')

if _VIZUALIZE:
    unwrapped_viz = cv2.remap(field.image, map_x_32, map_y_32, cv2.INTER_CUBIC)

unwrapped_bin = cv2.remap(bin_field, map_x_32, map_y_32, cv2.INTER_CUBIC)

digits_for_recog = []
digits_for_recog_coords = []
for i_row in range(_GRID_LINES - 1):
    for i_col in range(_GRID_LINES - 1):
        recognize_side = 28
        assert cell_side >= recognize_side
        margin_1 = (cell_side - recognize_side) // 2
        margin_2 = (cell_side - recognize_side) - margin_1
        digit_img = unwrapped_bin[
                    i_row * cell_side + margin_1:(i_row + 1) * cell_side - margin_2,
                    i_col * cell_side + margin_1:(i_col + 1) * cell_side - margin_2
                    ]
        digit_img[0, :] = 0
        digit_img[:, 0] = 0
        digit_img[recognize_side - 1, :] = 0
        digit_img[:, recognize_side - 1] = 0
        avg_intensity = np.sum(digit_img) / recognize_side / recognize_side
        if avg_intensity > 10:
            digits_for_recog.append(digit_img[np.newaxis, :, :])
            digits_for_recog_coords.append((i_row, i_col))

digits_for_recog = np.vstack(digits_for_recog)
labels = recognize_digits(digits_for_recog)
unsolved_field = np.zeros(shape=(9, 9), dtype=np.uint8)
for i, (i_row, i_col) in enumerate(digits_for_recog_coords):
    unsolved_field[i_row, i_col] = labels[i]
    # if _VIZUALIZE:
    #     label = str(labels[i])
    #     cv2.putText(unwrapped_viz, label,
    #                 org=(i_col * cell_side, (i_row + 1) * cell_side),
    #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=0.5,
    #                 color=(0, 255, 0),
    #                 lineType=2)

solved_field = solve(unsolved_field)
assert solved_field is not None

for i_row in range(9):
    for i_col in range(9):
        if unsolved_field[i_row, i_col] == 0:
            cell_side = field.side // 9
            text = str(solved_field[i_row, i_col])
            font_scale = 0.8
            (text_w, text_h), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                                                         thickness=1)
            if _VIZUALIZE:
                margin_x = (cell_side - text_w) // 2
                margin_y = (cell_side - text_h) // 2
                cv2.putText(unwrapped_viz, text,
                            org=(
                            i_col * cell_side + margin_x, (i_row + 1) * cell_side - margin_y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale,
                            thickness=1,
                            color=(0, 255, 0, 255),
                            lineType=cv2.LINE_AA)

# grid_x, grid_y = np.mgrid[0:field.side + field.margin * 2, 0:field.side + field.margin * 2]
# grid_z = griddata(np.array(src_points), np.array(dst_points), (grid_x, grid_y), method='cubic')
# map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(field.side + field.margin * 2, field.side + field.margin * 2)
# map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(field.side + field.margin * 2, field.side + field.margin * 2)
# map_x_32 = map_x.astype('float32')
# map_y_32 = map_y.astype('float32')
# wrapped_viz = cv2.remap(unwrapped_viz, map_x_32, map_y_32, cv2.INTER_CUBIC)


print((time.time() - t) * 1000)

if _VIZUALIZE:
    show_image("unwrapped", unwrapped_viz)
    # show_image("wrapped back", wrapped_viz)
    show_image("field_viz", field_viz, 700)

if _VIZUALIZE:
    wait_windows()
