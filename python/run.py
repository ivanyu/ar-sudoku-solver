#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np

import sudoku
from sudoku.solver import load_image, show_image, cut_out_field, clean_image, \
    binarize_field, \
    enforce_grid_simple, find_digit_bounding_boxes, extract_digits_from_cells, draw_overlay, \
    detect_grid_points, \
    warp_field, enforce_grid_detected

assert cv2.useOptimized()


def process(image: np.ndarray):
    time_start = time.time()

    image = clean_image(image)
    show_image("cleaned", image)

    field, corners, perspective_transform_matrix = cut_out_field(image)
    assert field is not None
    show_image("field", field.image)

    bin_field = binarize_field(field)
    show_image("field-bin", bin_field.image)

    # EXPERIMENTAL
    grid = detect_grid_points(bin_field)
    imgx = cv2.cvtColor(bin_field.image, cv2.COLOR_GRAY2BGR)
    # overlay_flat = np.zeros(shape=(field.image.shape[0], field.image.shape[1], 4), dtype=np.uint8)

    # for x, y in grid.reshape((-1, 2)):
    #     cv2.circle(imgx, (x, y), 2, (0, 0, 255), -1)
    #     pass

    # for i_row in range(9):
    #     for i_col in range(9):
    #         top_left_x, top_left_y = grid[i_row, i_col]
    #         bottom_right_x, bottom_right_y = grid[i_row + 1, i_col + 1]
    #         cell_image = bin_field.image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    #
    #         cell_side_h = bottom_right_y - top_left_y
    #         cell_side_w = bottom_right_x - top_left_x
    #         contours, _ = cv2.findContours(cell_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #         bb = None
    #         for contour in contours:
    #             x, y, w, h = cv2.boundingRect(contour)
    #             # Filter out too big and too small.
    #             if h > cell_side_h * 0.9 or w > cell_side_w * 0.9:
    #                 continue
    #             if w < cell_side_w * 0.2:
    #                 continue
    #             if h < cell_side_h * 0.5:
    #                 continue
    #             bb = sudoku.BoundingBox(x, y, w, h)
    #             break
    #         if bb is not None:
    #             cv2.rectangle(
    #                 overlay_flat,
    #                 (top_left_x + bb.x, top_left_y + bb.y),
    #                 (top_left_x + bb.x + bb.w, top_left_y + bb.y + bb.h),
    #                 color=(255, 0, 255, 255),
    #                 thickness=1,
    #                 lineType=cv2.LINE_AA)
    #
    #     pass
    # show_image("imgx", imgx)
    # show_image("overlay_flat", overlay_flat)
    # # show_image("overlay_mask", overlay_mask)
    #
    # out = field.image
    # alpha = overlay_flat[:, :, 3] / 255.0
    # out[:, :, 0] = (1. - alpha) * out[:, :, 0] + alpha * overlay_flat[:, :, 0]
    # out[:, :, 1] = (1. - alpha) * out[:, :, 1] + alpha * overlay_flat[:, :, 1]
    # out[:, :, 2] = (1. - alpha) * out[:, :, 2] + alpha * overlay_flat[:, :, 2]
    #
    # show_image("overlaed_field", out)
    #
    # overlay = cv2.warpPerspective(
    #     overlay_flat,
    #     perspective_transform_matrix,
    #     (image.shape[1], image.shape[0]),
    #     flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
    # )
    # show_image("overlay", overlay)

    # enforce_grid_simple(bin_field)
    enforce_grid_detected(bin_field, grid)
    # for i_row in range(9):
    #     for i_col in range(9):
    #         array = grid[i_row:i_row+2, i_col:i_col+2, :].reshape(-1, 2)
    #         array[[2, 3]] = array[[3, 2]]
    #         cv2.polylines(bin_field.image, [array], isClosed=True, color=255, thickness=1)
    show_image("field-bin-enforced_grid", bin_field.image)

    digit_bounding_boxes = find_digit_bounding_boxes(bin_field)

    if sudoku.DISPLAY:
        num_viz = field.image.copy()
        cell_side = bin_field.side // 9
        for bb in digit_bounding_boxes:
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

    number_patterns = []
    for digit in range(10):
        text = str(digit)
        (w, h), baseline = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            thickness=1)
        pattern = np.zeros(shape=(h + baseline // 2, w), dtype=np.uint8)
        cv2.putText(pattern,
                    text,
                    org=(0, h + baseline // 4),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=255,
                    thickness=1)
        number_patterns.append(pattern)
        show_image(text, pattern)
    pattern_shape = number_patterns[0].shape
    for pattern in number_patterns:
        assert pattern.shape == pattern_shape

    digits = extract_digits_from_cells(bin_field, digit_bounding_boxes)
    for i, digit in enumerate(digits):
        if digit is None:
            continue

        i_row = i // 9
        i_col = i % 9

        digit = cv2.copyMakeBorder(digit, 3, 2, 0, 0, cv2.BORDER_CONSTANT)
        scale_coeff = float(pattern_shape[0]) / digit.shape[0]
        digit = cv2.resize(digit, dsize=(int(digit.shape[1] * scale_coeff), pattern_shape[0]))
        h_diff = pattern_shape[1] - digit.shape[1]
        digit = cv2.copyMakeBorder(digit, 0, 0, h_diff // 2, h_diff - h_diff // 2, cv2.BORDER_CONSTANT)
        assert digit.shape == pattern.shape

        max_score = float("-inf")
        max_score_digit = -1
        for pattern_digit, pattern in enumerate(number_patterns):
            assert pattern.shape == digit.shape
            r = cv2.matchTemplate(digit, pattern, cv2.TM_CCOEFF)
            r = r[0, 0]
            if r > max_score:
                max_score = r
                max_score_digit = pattern_digit
        print(max_score_digit)
    # for pattern in number_patterns:


    # pattern = np.zeros(shape=(30, 30), dtype=np.uint8)
    # cv2.putText(pattern, "1",
    #             org=(0, 30),
    #             fontFace=cv2.FONT_HERSHEY_TRIPLEX,
    #             fontScale=1,
    #             color=255,
    #             thickness=2)


    print('ms per frame:', (time.time() - time_start) * 1000)

    return digits


# image = load_image("../images/big-numbers.jpg")
image = load_image("../images/sudoku.jpg")
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
