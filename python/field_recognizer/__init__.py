#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import griddata

from border_mapper import BorderMapper
from digit_recognizer_2 import create_recognizer
from sudoku import Field
from sudoku.solver import cut_out_field, perspective_transform_contour, find_corners, extract_subcontour


_GRID_LINES = 10

_VIZUALIZE = False
_SAVE_DIGIT_IMAGES = True

if _VIZUALIZE:
    from utils import show_image, wait_windows


def recognize_field(image: np.ndarray):
    # Extract the field, its contour and corners.
    field, field_contour, _, perspective_transform_matrix = cut_out_field(image)
    field_gray = Field(
        _gray_image(field.image),
        field.side,
        field.margin
    )

    field_bin_img = cv2.adaptiveThreshold(
        field_gray.image, maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=17,
        C=11)
    if _VIZUALIZE:
        show_image("field_bin", field_bin_img)
    field_bin = Field(
        field_bin_img,
        field.side,
        field.margin
    )

    _erase_numbers(field_gray, field_bin)

    grid_points = _find_grid_points(field_gray, field_contour, perspective_transform_matrix)
    unwrapped_field_img = _upwrap_field(field_bin, grid_points)
    if _VIZUALIZE:
        show_image("unwrapped_field_img", unwrapped_field_img)

    return _recognize_field(unwrapped_field_img, field_gray.ideal_cell_side())


def _gray_image(image: np.ndarray) -> np.ndarray:
    field_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    return field_gray


def _erase_numbers(field_gray: Field, field_bin: Field):
    contours, _ = cv2.findContours(field_bin.image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cell_side = field_gray.ideal_cell_side()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > cell_side * 0.8 or w < 0.2 * cell_side:
            continue
        if h > cell_side * 0.9 or h < 0.2 * cell_side:
            continue
        # cv2.rectangle(field_viz, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=-1)
        # cv2.rectangle(field_viz, (x, y), (x + w, y + h), color=(0, 0, 0), thickness=-1)
        # cv2.drawContours(field_viz, [contour], 0, color=(0, 255, 0), thickness=-1)
        cv2.drawContours(field_gray.image, [contour], 0, color=255, thickness=2)
        cv2.drawContours(field_gray.image, [contour], 0, color=255, thickness=-1)

    if _VIZUALIZE:
        show_image("field_gray no numbers", field_gray.image)


def _find_grid_points(field_gray: Field, field_contour: np.ndarray, perspective_transform_matrix: np.ndarray) -> np.ndarray:
    top_border, left_border = _get_top_and_left_borders(field_contour, perspective_transform_matrix)
    horizontal_lines = _find_horizontal_lines(field_gray, BorderMapper(left_border))
    vertical_lines = _find_vertical_lines(field_gray, BorderMapper(top_border))
    horizontal_lines_masks, vertical_lines_masks = _get_line_masks(horizontal_lines, vertical_lines, field_gray.image.shape[0])

    # # TODO ? intersect one horizontal with all vertical
    intersection = np.zeros(shape=(field_gray.image.shape[0], field_gray.image.shape[1]), dtype=np.uint8)
    grid_points = np.zeros(shape=(_GRID_LINES, _GRID_LINES, 2), dtype=np.uint32)

    for i_row in range(_GRID_LINES):
        for i_col in range(_GRID_LINES):
            np.bitwise_and(horizontal_lines_masks[i_row], vertical_lines_masks[i_col], out=intersection)
            intersection_points = np.argwhere(intersection == 255)

            # There should not be more than several intersection points:
            # one is the default, 2 might be due to the overlapping of steps in lines.
            assert 1 <= intersection_points.shape[0] <= 2
            grid_points[i_row, i_col] = np.flip(intersection_points[0])  # put x before y

            # if _VIZUALIZE:
            #     y, x = intersection_points[0]
            #     cv2.circle(field_viz, (x, y), radius=1, color=(100, 100, 255), thickness=-1)
            #     cv2.putText(field_viz,
            #                 text=f"{i_row},{i_col}",
            #                 org=(x, y),
            #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                 fontScale=0.3,
            #                 color=(0, 0, 255))

    return grid_points


def _get_top_and_left_borders(field_contour: np.ndarray, perspective_transform_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    # Swap x and y.
    top_border = np.flip(top_border, axis=1)
    # right_border = extract_border(bottom_right_idx, top_right_idx)
    # bottom_border = extract_border(bottom_left_idx, bottom_right_idx)
    left_border = extract_subcontour(transformed_field_contour, top_left_idx, bottom_left_idx)
    return top_border, left_border


def _find_vertical_lines(field_gray: Field, top_border_mapper: BorderMapper) -> List[List[Tuple[int, int]]]:
    grad_x = cv2.Sobel(field_gray.image, ddepth=cv2.CV_64F, dx=2, dy=0, ksize=7, scale=1, delta=0,
                       borderType=cv2.BORDER_DEFAULT)
    np.clip(grad_x, a_min=0, a_max=grad_x.max(), out=grad_x)
    grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    if _VIZUALIZE:
        show_image("grad_x", grad_x)

    vertical_lines = _detect_lines(
        cv2.rotate(grad_x, cv2.ROTATE_90_COUNTERCLOCKWISE),
        top_border_mapper,
        field_gray.margin,
        field_gray.margin + field_gray.side,
        field_gray.ideal_cell_side()
    )
    assert len(vertical_lines) == _GRID_LINES
    return vertical_lines


def _find_horizontal_lines(field_gray: Field, left_border_mapper: BorderMapper) -> List[List[Tuple[int, int]]]:
    grad_y = cv2.Sobel(field_gray.image, ddepth=cv2.CV_64F, dx=0, dy=2, ksize=7, scale=1, delta=0,
                       borderType=cv2.BORDER_DEFAULT)
    np.clip(grad_y, a_min=0, a_max=grad_y.max(), out=grad_y)
    grad_y = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    if _VIZUALIZE:
        show_image("grad_y", grad_y)

    horizontal_lines = _detect_lines(
        grad_y,
        left_border_mapper,
        field_gray.margin,
        field_gray.margin + field_gray.side,
        field_gray.ideal_cell_side()
    )
    assert len(horizontal_lines) == _GRID_LINES
    return horizontal_lines


def _detect_lines(work_image: np.ndarray, border_mapper: BorderMapper, left_limit: int, right_limit: int, cell_side: int) -> List[List[Tuple[int, int]]]:
    work_image_blur = cv2.GaussianBlur(work_image, (1, 25), 0)

    offset = cell_side // 6
    # offset = 0
    step = 1
    win_w = cell_side // 4

    detected_windows = []
    for y in range(left_limit, right_limit + 1, step):
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
        line = _detect_line(work_image, x, y, win_h, win_w, right_limit)

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
        x = work_image.shape[1]
        y = int(round(a * x + b))
        line.append((x, y))

        result.append(line)
        # break
    return result


def _detect_line(image, start_x, start_y, win_h, win_w, right_limit) -> Optional[List[Tuple[int, int]]]:
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

        avg = _get_average_y_over_window(image, win_x, win_y, current_win_w, win_h)
        if avg is None:
            win_y -= win_h // 2
            avg = _get_average_y_over_window(image, win_x, win_y, current_win_w, win_h * 2)

        if avg is None:
            lost_line = True

        win_x = win_x + current_win_w
        win_y = win_y + (avg - win_h // 2)
        result.append((win_x, win_y + win_h // 2))

    if not lost_line:
        return result
    else:
        return None


def _get_average_y_over_window(image, win_x: int, win_y: int, win_w: int, win_h: int) -> Optional[int]:
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


def _get_line_masks(horizontal_lines: List[List[Tuple[int, int]]], vertical_lines: List[List[Tuple[int, int]]],
                    image_side: int) -> Tuple[np.ndarray, np.ndarray]:
    mask_shape = (_GRID_LINES, image_side, image_side)
    vertical_lines_masks = np.zeros(shape=mask_shape, dtype=np.uint8)
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

    # if _VIZUALIZE:
    #     cv2.rotate(field_viz, cv2.ROTATE_90_CLOCKWISE, dst=field_viz)

    horizontal_lines_masks = np.zeros(shape=mask_shape, dtype=np.uint8)
    for i, line in enumerate(horizontal_lines):
        poly = [np.array(line, np.int32)]
        # if _VIZUALIZE:
        #     for x, y in line:
        #         cv2.circle(field_viz, (x, y), 0, (255, 255, 0), 2)
        #     cv2.polylines(field_viz, poly, isClosed=False, color=(255, 255, 0), thickness=1)
        cv2.polylines(horizontal_lines_masks[i], poly, isClosed=False, color=255, thickness=1)

    return horizontal_lines_masks, vertical_lines_masks


def _upwrap_field(field: Field, grid_points: np.ndarray) -> np.ndarray:
    src_points = []
    dst_points = []
    cell_side = field.ideal_cell_side()
    for i_row in range(_GRID_LINES):
        for i_col in range(_GRID_LINES):
            src_points.append(np.flip(grid_points[i_row, i_col]))
            dst_points.append((i_row * cell_side, i_col * cell_side))

    grid_x, grid_y = np.mgrid[0:field.side, 0:field.side]
    grid_z = griddata(np.array(dst_points), np.array(src_points), (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(field.side, field.side).astype(np.float32)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(field.side, field.side).astype(np.float32)

    # if _VIZUALIZE:
    #     unwrapped_viz = cv2.remap(field.image, map_x, map_y, cv2.INTER_CUBIC)

    unwrapped = cv2.remap(field.image, map_x, map_y, cv2.INTER_CUBIC)
    return unwrapped


def _recognize_field(unwrapped_field_img: np.ndarray, cell_side: int) -> np.ndarray:
    recognizer = create_recognizer()

    digits_to_recognize = []
    digits_to_recognize_2 = []
    digits_to_recognize_3 = []
    digits_to_recognize_coords = []
    for i_row in range(_GRID_LINES - 1):
        for i_col in range(_GRID_LINES - 1):
            recognize_side = 28
            assert cell_side >= recognize_side
            margin_1 = (cell_side - recognize_side) // 2 + 1
            margin_2 = (cell_side - recognize_side) - margin_1
            digit_img = unwrapped_field_img[
                        i_row * cell_side + margin_1:(i_row + 1) * cell_side - margin_2,
                        i_col * cell_side + margin_1:(i_col + 1) * cell_side - margin_2
                        ]
            # digit_img[0, :] = 0
            # digit_img[:, 0] = 0
            # digit_img[recognize_side - 1, :] = 0
            # digit_img[:, recognize_side - 1] = 0

            # line_width = 3
            # digit_img = unwrapped_field_img[
            #     i_row * cell_side + line_width:(i_row + 1) * cell_side - line_width + 1,
            #     i_col * cell_side + line_width:(i_col + 1) * cell_side - line_width + 1
            # ]
            # cv2.imwrite(f"{i_row}-{i_col}-before.jpg", digit_img)

            # digit_img[:line_width, :] = 0
            # digit_img[cell_side - line_width:, :] = 0
            # digit_img[:, :line_width] = 0
            # digit_img[:, cell_side - line_width:] = 0
            # digit_img = cv2.copyMakeBorder(digit_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
            # target_h = 28
            # target_w = 28
            # scale = float(target_h) / digit_img.shape[0]
            # digit_img = cv2.resize(digit_img, dsize=(int(digit_img.shape[1] * scale), target_h))
            # w_diff = target_w - digit_img.shape[1]
            # assert w_diff >= 0
            # digit_img = cv2.copyMakeBorder(digit_img, 0, 0, w_diff // 2, w_diff - (w_diff // 2), cv2.BORDER_CONSTANT)
            # cv2.imwrite(f"{i_row}-{i_col}-after.jpg", digit_img)

            avg_intensity = np.sum(digit_img) / recognize_side / recognize_side

            if _SAVE_DIGIT_IMAGES:
                import os
                import random
                import string
                dir = "data/xx"
                os.makedirs(dir, exist_ok=True)

                name = ''.join(random.choice(string.ascii_lowercase) for i in range(9))
                name = f"{i_row}-{i_col}"
                if avg_intensity > 10:
                    cv2.imwrite(os.path.join(dir, f"{name}.jpg"), digit_img)
                else:
                    os.makedirs(os.path.join(dir, "empty"), exist_ok=True)
                    cv2.imwrite(os.path.join(dir, "empty", f"{name}.jpg"), digit_img)

            if avg_intensity > 10:
                digits_to_recognize.append(digit_img)
                digits_to_recognize_coords.append((i_row, i_col))

                from PIL import Image, ImageFilter, ImageDraw
                import torchvision as tv
                img = Image.fromarray(digit_img)
                img = tv.transforms.functional.affine(img, angle=0, translate=(0, 0), scale=0.9,
                                                      shear=0, resample=Image.BICUBIC)
                # draw = ImageDraw.Draw(img)
                # print(draw)
                # r = img.filter(ImageFilter.GaussianBlur(radius=1))
                digit_img_2 = np.array(img)

                # rows, cols = digit_img.shape
                # gauss = np.random.normal(0, 3, (rows, cols))
                # gauss = gauss.reshape(rows, cols)
                # digit_img_3 = digit_img.astype(np.float) + gauss
                # digit_img_3 = cv2.normalize(digit_img_3, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                # digit_img_3 = cv2.threshold(digit_img, 150, 255, cv2.THRESH_BINARY)[1]

                img = tv.transforms.functional.affine(img, angle=0, translate=(0, 0), scale=0.8,
                                                      shear=0, resample=Image.BICUBIC)
                # draw = ImageDraw.Draw(img)
                # print(draw)
                # r = img.filter(ImageFilter.GaussianBlur(radius=1))
                digit_img_3 = np.array(img)

                # digit_img_2 = cv2.copyMakeBorder(cv2.resize(digit_img, (24, 24)), 2, 2, 2, 2, cv2.BORDER_CONSTANT)
                # digit_img_2 = cv2.morphologyEx(digit_img_2, cv2.MORPH_CLOSE,
                #                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
                #
                # digit_img_3 = cv2.resize(digit_img, (32, 32))[2:30, 2:30]
                # digit_img_3 = cv2.morphologyEx(digit_img_3, cv2.MORPH_OPEN,
                #                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))

                if _SAVE_DIGIT_IMAGES:
                    import os
                    import random
                    import string
                    dir = "data/xx"
                    os.makedirs(dir, exist_ok=True)

                    name = ''.join(random.choice(string.ascii_lowercase) for i in range(9))
                    name = f"{i_row}-{i_col}"
                    cv2.imwrite(os.path.join(dir, f"{name}.jpg"), digit_img)
                    cv2.imwrite(os.path.join(dir, f"{name}.png"), digit_img)
                    cv2.imwrite(os.path.join(dir, f"{name}-2.jpg"), digit_img_2)
                    cv2.imwrite(os.path.join(dir, f"{name}-3.jpg"), digit_img_3)

                # digits_to_recognize_2.append(digit_img_2[np.newaxis, :, :])
                digits_to_recognize_2.append(digit_img_2)
                # digits_to_recognize_3.append(digit_img_3[np.newaxis, :, :])
                digits_to_recognize_3.append(digit_img_3)

    # digits_to_recognize = np.vstack(digits_to_recognize)
    # digits_to_recognize_2 = np.vstack(digits_to_recognize_2)
    # digits_to_recognize_3 = np.vstack(digits_to_recognize_3)

    # special1 = recognizer(digits_to_recognize[22][np.newaxis, :, :])
    # import os
    # cv2.imwrite("tmp.jpg", digits_to_recognize[22])
    #
    # from PIL import Image
    # img = Image.open("tmp.jpg")
    # import torchvision as tv
    # tensor = tv.transforms.ToTensor()(img).unsqueeze(1).float()
    # from digit_recognizer_2 import DigitRecognizer2
    # _digit_recognizer: DigitRecognizer2 = DigitRecognizer2()
    # _path = os.path.join(
    #     os.path.dirname(os.path.dirname(__file__)),
    #     "model-ft.pth"
    # )
    # import torch
    # _digit_recognizer.load_state_dict(torch.load(_path))
    # _digit_recognizer.eval()
    # special2 = torch.max(_digit_recognizer(tensor).data, dim=1)[1] + 1

    labels, outputs = recognizer(digits_to_recognize)
    labels_2, outputs_2 = recognizer(digits_to_recognize_2)
    labels_3, outputs_3 = recognizer(digits_to_recognize_3)
    print(labels)
    print(labels_2)
    print(labels_3)
    for i in range(len(labels)):
        if labels[i] == labels_2[i] or labels[i] == labels_3[i]:
            continue
        elif labels_2[i] == labels_3[i]:
            labels[i] = labels_2[i]
        else:
            assert False, "Can't recognize"

    recognized_field = np.zeros(shape=(9, 9), dtype=np.uint8)
    for i, (i_row, i_col) in enumerate(digits_to_recognize_coords):
        recognized_field[i_row, i_col] = labels[i]
        # if _VIZUALIZE:
        #     label = str(labels[i])
        #     cv2.putText(unwrapped_viz, label,
        #                 org=(i_col * cell_side, (i_row + 1) * cell_side),
        #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=0.5,
        #                 color=(0, 255, 0),
        #                 lineType=2)
    return recognized_field


if __name__ == "__main__":
    from sudoku.solver import load_image
    from utils import scale_image
    image_files = [
        "../images/big-numbers.jpg",
        "../images/slightly_blurry.jpg",
        "../images/sudoku.jpg",
        "../images/sudoku-rotated.jpg",
        "../images/sudoku-1.jpg",
        "../images/sudoku-2.jpg",
        "../images/sudoku-2-rotated.jpg",
    ]

    image_files = ["../images/sudoku-rotated.jpg"]

    for f in image_files:
        image = load_image(f)
        image = scale_image(image, 640)
        recognized_field = recognize_field(image)
        print(recognized_field)
    if _VIZUALIZE:
        wait_windows()
