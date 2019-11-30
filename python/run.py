#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np

import sudoku
from sudoku.solver import load_image, show_image, cut_out_field, clean_image, \
    binarize_field, \
    enforce_grid, find_number_bounding_boxes, assign_number_bounding_boxes_to_cells, draw_overlay, detect_grid_points

assert cv2.useOptimized()


def process(image: np.ndarray):
    time_start = time.time()

    image = clean_image(image)
    show_image("cleaned", image)

    field, corners = cut_out_field(image)
    assert field is not None
    show_image("field", field.image)

    bin_field = binarize_field(field)
    show_image("field-bin", bin_field.image)

    # EXPERIMENTAL
    detect_grid_points(bin_field)

    enforce_grid(bin_field)
    show_image("field-bin-enforced_grid", bin_field.image)

    number_bounding_boxes = find_number_bounding_boxes(bin_field)

    if sudoku.DISPLAY:
        num_viz = field.image.copy()
        cell_side = bin_field.side // 9
        for bb in number_bounding_boxes:
            x_cell = (bb.x - bin_field.margin) // cell_side
            y_cell = (bb.y - bin_field.margin) // cell_side
            cv2.rectangle(num_viz, (bb.x, bb.y), (bb.x + bb.w, bb.y + bb.h), (255, 0, 255), 2)
            cv2.putText(num_viz, f"{x_cell}-{y_cell}",
                        org=(bb.x, bb.y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        lineType=2)
        show_image("num_viz", num_viz)
        draw_overlay(
            image,
            num_viz,
            corners,
            field.side, field.margin
        )

    number_bounding_boxes_by_cells = assign_number_bounding_boxes_to_cells(bin_field, number_bounding_boxes)

    print('ms per frame:', (time.time() - time_start) * 1000)

    return number_bounding_boxes_by_cells


image = load_image("../images/big-numbers.jpg")
show_image("original_resized", image)
process(image)
print()

if False:
    images_and_expected = {
        "../images/sudoku.jpg": {
            0, 4, 8, 10, 12, 14, 16, 20, 22,
            24, 28, 30, 32, 34, 36, 38, 40, 42,
            44, 46, 48, 50, 56, 58, 64, 66, 68,
            70, 72, 76, 80},
        "../images/sudoku-1.jpg": {
            3, 5, 6, 9, 11, 17, 23, 25, 28,
            31, 34, 35, 36, 44, 45, 46, 49,
            52, 55, 57, 63, 69, 71, 74, 75, 77},
        "../images/sudoku-2.jpg": {
            1, 2, 3, 9, 11, 13, 17, 18, 21, 22,
            24, 27, 37, 41, 45, 47, 52, 53, 58,
            61, 64, 66, 71, 72, 78},
        "../images/big-numbers.jpg": {
            5, 6, 9, 12, 16, 18, 24, 29, 30, 31,
            35, 36, 39, 40, 41, 44, 45, 49, 50,
            51, 56, 62, 64, 68, 71, 74, 75},
    }
    images_and_expected["../images/sudoku-rotated.jpg"] = images_and_expected["../images/sudoku.jpg"]
    images_and_expected["../images/sudoku-2-rotated.jpg"] = images_and_expected["../images/sudoku-2.jpg"]

    for filename in images_and_expected:
        print(filename)
        image = load_image(filename)
        number_bounding_boxes_by_cells = process(image)
        for i in range(len(number_bounding_boxes_by_cells)):
            assert i not in images_and_expected[filename] or number_bounding_boxes_by_cells[i] is not None

if sudoku.DISPLAY:
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
