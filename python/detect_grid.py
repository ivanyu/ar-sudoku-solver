#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import Tuple

import cv2
import numpy as np

from sudoku import Field
from sudoku.solver import load_image, cut_out_field, show_image, clean_image, \
    find_digit_bounding_boxes


def find_number_mask(bin_field: Field):
    number_mask = np.full(bin_field.image.shape, 255, np.uint8)
    for x, y, w, h in find_digit_bounding_boxes(bin_field):
        cv2.rectangle(number_mask, (x, y), (x + w, y + h), 0, -1)
    return number_mask


def find_grid(field: Field):
    cell_side = field.side // 9
    min_point_distance_sq: float = (cell_side * 0.8) ** 2

    best_points = []

    def add_point(point) -> bool:
        for p in best_points:
            if line_len_sq(p, point) < min_point_distance_sq:
                return False
        best_points.append(point)
        return True

    for tpl_func, expected_to_find in [
        (_tpl_cross, 64),
        (_tpl_top_left, 17),
        (_tpl_bottom_right, 17)
    ]:
        (tpl, tpl_side) = tpl_func(cell_side)
        res = cv2.matchTemplate(field.image, tpl, cv2.TM_CCOEFF)
        sorted_res = np.argsort(res, axis=None)
        sorted_rows, sorted_cols = np.unravel_index(sorted_res, res.shape)
        max_points_to_check = sorted_res.shape[0] // 100

        found = 0
        for i in range(-1, -max_points_to_check, -1):
            x = sorted_cols[i]
            y = sorted_rows[i]
            if tpl_func == _tpl_cross:
                x += tpl_side // 2
                y += tpl_side // 2
            elif tpl_func == _tpl_bottom_right:
                x += tpl_side
                y += tpl_side
            if add_point((x, y)):
                found += 1
                if found == expected_to_find:
                    break

    add_point((field.margin + field.side, field.margin))
    add_point((0 + field.margin, field.margin + field.side))

    best_points = sorted(best_points, key=lambda p: (p[1] // cell_side, p[0]))
    return best_points


def _tpl_cross(cell_side: int) -> Tuple[np.ndarray, int]:
    tpl_side = int(cell_side * 1.2)
    tpl = np.ones((tpl_side, tpl_side), np.uint8)
    cv2.line(tpl,
             (0, tpl_side // 2),
             (tpl_side, tpl_side // 2),
             color=255, thickness=1)
    cv2.line(tpl,
             (tpl_side // 2, 0),
             (tpl_side // 2, tpl_side),
             color=255, thickness=1)
    return tpl, tpl_side


def _tpl_top_left(cell_side: int) -> Tuple[np.ndarray, int]:
    tpl_side = int(cell_side * 0.9)
    tpl = np.ones((tpl_side, tpl_side), np.uint8)
    cv2.line(tpl,
             (0, 0),
             (0, tpl_side),
             color=255, thickness=1)
    cv2.line(tpl,
             (0, 0),
             (tpl_side, 0),
             color=255, thickness=1)
    return tpl, tpl_side


def _tpl_bottom_right(cell_side: int) -> Tuple[np.ndarray, int]:
    tpl_side = int(cell_side * 0.9)
    tpl = np.ones((tpl_side, tpl_side), np.uint8)
    cv2.line(tpl,
             (tpl_side - 1, tpl_side - 1),
             (0, tpl_side - 1),
             color=255, thickness=1)
    cv2.line(tpl,
             (tpl_side - 1, tpl_side - 1),
             (tpl_side - 1, 0),
             color=255, thickness=1)
    return tpl, tpl_side


def line_len_sq(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


tracker = cv2.TrackerCSRT_create()

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../images/VID_20191201_104129.mp4")
found = False
while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    scale = 0.4
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # cv2.imshow('frame', frame)
    frame = clean_image(frame)
    # cv2.imshow('frame-clean', frame)

    out = frame.copy()
    if not found:
        field, corners, perspective_transform_matrix = cut_out_field(frame)
        # tracker.init(frame, (corners.top_left[0], corners.top_left[1], corners.bottom_right[0], corners.bottom_right[1]))
        # print(corners)
        cv2.rectangle(out, tuple(corners.top_left), tuple(corners.bottom_right), 0, 1)
        # found = True
    else:
        success, boxes = tracker.update(frame)
        box = boxes[0]
        print(boxes)
        tl = (int(boxes[0]), int(boxes[1]))
        br = (int(boxes[2]), int(boxes[3]))
        cv2.rectangle(out, tl, br, 0, 1)

        # if field is not None and field.side >= 100:
        #     bin_field = binarize_field(field)
        #     enforce_grid(bin_field)
        #     number_mask = find_number_mask(bin_field)
        #     field_no_numbers_image = cv2.bitwise_and(bin_field.image, number_mask)
        #     # cv2.imshow('field_no_numbers_image', field_no_numbers_image)
        #     field_no_numbers = Field(field_no_numbers_image, bin_field.side, bin_field.margin)
        #     grid = find_grid(field_no_numbers)
        #
        #     overlay = np.zeros([field.image.shape[0], field.image.shape[1], 3], dtype=np.uint8)
        #     for i, point in enumerate(grid):
        #         # cv2.putText(
        #         #     overlay,
        #         #     str(i),
        #         #     org=point,
        #         #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         #     fontScale=0.5,
        #         #     color=(0, 255, 0),
        #         #     lineType=2)
        #         cv2.circle(overlay, point, radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
        #     overlayed = draw_overlay(
        #         frame,
        #         overlay,
        #         corners,
        #         field.side, field.margin
        #     )
        #
    cv2.imshow('in', frame)
    cv2.imshow('out', out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_time_ms = (time.time() - frame_start) * 1000
    print(f"Time per frame: {frame_time_ms}, FPS: {1000 // frame_time_ms}")

cap.release()
cv2.destroyAllWindows()

# image = load_image("../images/sudoku.jpg")
# show_image("original_resized", image)
#
# image = clean_image(image)
# show_image("cleaned", image)
#
# field, corners = cut_out_field(image)
# assert field is not None
# show_image("field", field.image)
#
# bin_field = binarize_field(field)
# show_image("field-bin", bin_field.image)
#
# enforce_grid(bin_field)
# show_image("field-bin-enforced_grid", bin_field.image)
#
# number_mask = find_number_mask(bin_field)
# field_no_numbers = cv2.bitwise_and(bin_field.image, number_mask)
# show_image("field_no_numbers", field_no_numbers)
#
# grid = find_grid(bin_field)
# # assert len(grid) == 100
# overlay = np.zeros([field.image.shape[0], field.image.shape[1], 3], dtype=np.uint8)
# grid_image = cv2.cvtColor(field_no_numbers, cv2.COLOR_GRAY2BGR)
# for i, point in enumerate(grid):
#     # cv2.putText(
#     #     grid_image,
#     #     str(i),
#     #     org=point,
#     #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#     #     fontScale=0.5,
#     #     color=(0, 255, 0),
#     #     lineType=2)
#     cv2.circle(grid_image, point, radius=3, color=(0, 255, 0), thickness=-1)
#     cv2.circle(overlay, point, radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
#
# overlayed = draw_overlay(
#     image,
#     overlay,
#     corners,
#     field.side, field.margin
# )
# show_image('Overlayed', overlayed)
#
# # show_image('Grid points', overlay)
#
# # warp_grid(image, grid, field_side // 9, field_margin)

# if sudoku.DISPLAY:
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# cv2.destroyAllWindows()
