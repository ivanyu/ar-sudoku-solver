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


def recognize_field(image: np.ndarray) -> np.ndarray:
    # Extract the field, its contour and corners.
    field, field_contour, _, perspective_transform_matrix = cut_out_field(image)
    field_gray = Field(
        _gray_image(field.image),
        field.side,
        field.margin
    )

    grid_points = _find_grid_points(field_gray, field_contour, perspective_transform_matrix)

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
    unwrapped_field_img = _upwrap_field(field_bin, grid_points)
    if _VIZUALIZE:
        show_image("unwrapped_field_img", unwrapped_field_img)

    if _VIZUALIZE:
        wait_windows()

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


def _find_grid_points(field_gray: Field, field_contour: np.ndarray, perspective_transform_matrix: np.ndarray) -> np.ndarray:
    top_border, right_border, bottom_border, left_border = _get_borders(field_contour, perspective_transform_matrix)

    horizontal_lines = _find_horizontal_lines(field_gray, top_border, bottom_border, left_border)
    vertical_lines = _find_vertical_lines(field_gray, top_border, right_border, left_border)
    horizontal_line_masks, vertical_line_masks = _get_line_masks(horizontal_lines, vertical_lines, field_gray.image.shape[0])

    # # TODO ? intersect one horizontal with all vertical
    intersection = np.zeros(shape=(field_gray.image.shape[0], field_gray.image.shape[1]), dtype=np.uint8)
    grid_points = np.zeros(shape=(_GRID_LINES, _GRID_LINES, 2), dtype=np.uint32)

    for i_row in range(_GRID_LINES):
        for i_col in range(_GRID_LINES):
            np.bitwise_and(horizontal_line_masks[i_row], vertical_line_masks[i_col], out=intersection)
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


def _get_borders(field_contour: np.ndarray, perspective_transform_matrix: np.ndarray) ->\
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return top_border, right_border, bottom_border, left_border


def _find_vertical_lines(field_gray: Field,
                         top_border: np.ndarray,
                         right_border: np.ndarray,
                         left_border: np.ndarray) -> List[List[Tuple[int, int]]]:
    grad_x = cv2.Sobel(field_gray.image, ddepth=cv2.CV_64F, dx=2, dy=0, ksize=7, scale=1, delta=0,
                       borderType=cv2.BORDER_DEFAULT)
    np.clip(grad_x, a_min=0, a_max=grad_x.max(), out=grad_x)
    grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    if _VIZUALIZE:
        show_image("grad_x", grad_x)

    grad_x_rot = cv2.rotate(grad_x, cv2.ROTATE_90_COUNTERCLOCKWISE)
    work_field = Field(
        grad_x_rot,
        field_gray.side,
        field_gray.margin
    )

    right_border_rot = _rotate_border(field_gray, right_border, change_order=False)
    left_border_rot = _rotate_border(field_gray, left_border, change_order=False)
    top_border_rot = _rotate_border(field_gray, top_border, change_order=True)
    vertical_lines = _find_lines(
        work_field,
        right_border_rot,
        left_border_rot,
        top_border_rot,
    )
    return vertical_lines


def _find_horizontal_lines(field_gray: Field,
                           top_border: np.ndarray,
                           bottom_border: np.ndarray,
                           left_border: np.ndarray) -> List[List[Tuple[int, int]]]:
    grad_y = cv2.Sobel(field_gray.image, ddepth=cv2.CV_64F, dx=0, dy=2, ksize=7, scale=1, delta=0,
                       borderType=cv2.BORDER_DEFAULT)
    np.clip(grad_y, a_min=0, a_max=grad_y.max(), out=grad_y)
    grad_y = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    if _VIZUALIZE:
        show_image("grad_y", grad_y)

    work_field = Field(
        grad_y,
        field_gray.side,
        field_gray.margin
    )

    horizontal_lines = _find_lines(
        work_field,
        top_border,
        bottom_border,
        left_border,
    )
    return horizontal_lines


def _rotate_border(field: Field, border: np.ndarray, change_order: bool) -> np.ndarray:
    border = np.copy(border)
    # Change points order so they go from the top left corner.
    if change_order:
        border = np.flip(border, axis=0)
    # Swap x and y.
    border = np.flip(border, axis=1)
    border[:, 1] = field.image.shape[0] - border[:, 1]
    return border


def _find_lines(work_field: Field,
                top_border: np.ndarray, bottom_border: np.ndarray, left_border: np.ndarray)\
        -> List[List[Tuple[int, int]]]:
    cell_side = work_field.ideal_cell_side()
    points_to_fit_on_extrapolation = 5

    _, work_image_thresh = cv2.threshold(work_field.image, 5, 255, cv2.THRESH_BINARY)
    work_field_thresh = Field(
        work_image_thresh,
        work_field.side,
        work_field.margin
    )

    lines = [[] for _ in range(10)]

    # Refine borders.
    for x, y in top_border:
        y = _refine_point(work_field.image, x, y, 2)
        if y is not None:
            lines[0].append((x, y))
    for x, y in bottom_border:
        y = _refine_point(work_field.image, x, y, 3)
        assert y is not None
        if y is not None:
            lines[9].append((x, y))

    horizonal_look_ahead = cell_side * 2

    first_point_behind = np.argmax(top_border[:, 0] >= top_border[0, 0] + horizonal_look_ahead) + 1
    current_line = top_border[:first_point_behind + 1]

    left_border_mapper = BorderMapper(left_border)

    for i in range(1, 9):
        current_line = _find_next_line(work_field_thresh, current_line, left_border_mapper, horizonal_look_ahead)

        # Due to noise and the border thickness, the first point is likely to be an outlier,
        # replacing it with an exptrapolation.
        x = 0
        y = _extrapolate_y(current_line[1:points_to_fit_on_extrapolation + 2], x)
        current_line[0] = (x, y)

        full_line = _continue_line(work_field_thresh, current_line)

        # Due to noise and the border thickness, the last point is likely to be an outlier,
        # replacing it with an exptrapolation.
        x = work_field.image.shape[1]
        y = _extrapolate_y(full_line[-points_to_fit_on_extrapolation - 1:-1], x)
        full_line[-1] = (x, y)

        lines[i] = full_line

    # Extrapolate borders.
    for line_i in [0, -1]:
        line = lines[line_i]

        x = 0
        y = _extrapolate_y(line[:points_to_fit_on_extrapolation + 1], x)
        line.insert(0, (x, y))

        x = work_field.image.shape[1]
        y = _extrapolate_y(line[-points_to_fit_on_extrapolation:], x)
        line.append((x, y))

    return lines


def _find_next_line(work_field_thresh: Field,
                    current_line: np.ndarray, left_border_mapper: BorderMapper, horizonal_look_ahead: int)\
        -> List[Tuple[int, int]]:
    cell_side = work_field_thresh.ideal_cell_side()

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

    def create_mask(x1_rel: int, y1_rel: int, x2_rel: int, y2_rel: int) -> np.ndarray:
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

        sub_img = work_field_thresh.image[y_off:mask.shape[0] + y_off, x_off:mask.shape[1] + x_off]
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

        y_int = _refine_point(work_field_thresh.image, x_int, y_int, 2)
        if y_int is not None:
           line.append((x_int, y_int))

        x += dx
        y += dy
    return line


def _continue_line(work_field_thresh: Field, line: List[Tuple[int, int]])\
        -> List[Tuple[int, int]]:
    line = list(line)
    for run in range(1000):  # effectively infinite
        tail = np.array(line[-10:])
        a, b = np.polyfit(tail[:, 0], tail[:, 1], 1)
        x, y = tail[-1]
        dx = 5
        dy = a * dx

        if x >= work_field_thresh.margin + work_field_thresh.side:
            break

        new_points = _continue_line_run(work_field_thresh, x, y, dx, dy)
        if len(new_points) == 0:
            break
        line += new_points
    return line


def _continue_line_run(work_field_thresh: Field, x: int, y: int, dx: float, dy: float)\
        -> List[Tuple[int, int]]:
    new_points = []
    skipped_steps = 0
    for _ in range(40):
        x += dx

        if x >= work_field_thresh.margin + work_field_thresh.side:
            break

        y += dy
        x_int = int(round(x))
        y_int = int(round(y))

        y_refined = _refine_point(work_field_thresh.image, x_int, y_int, 1)
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


def _refine_point(work_image: np.ndarray, x: int, y: int, win_h_half: int) -> Optional[int]:
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


def _extrapolate_y(points: List[Tuple[int, int]], x: int) -> int:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    a, b = np.polyfit(xs, ys, 1)
    y = int(round(a * x + b))
    return y


def _get_line_masks(horizontal_lines: List[List[Tuple[int, int]]], vertical_lines: List[List[Tuple[int, int]]],
                    image_side: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    vertical_line_masks = []
    for i, line in enumerate(vertical_lines):
        poly = [np.array(line, np.int32)]
        mask = np.zeros(shape=(image_side, image_side), dtype=np.uint8)
        cv2.polylines(mask, poly, isClosed=False, color=255, thickness=1)
        cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE, dst=mask)
        # Prepending is a part of rotation.
        vertical_line_masks.insert(0, mask)

    horizontal_line_masks = []
    for i, line in enumerate(horizontal_lines):
        poly = [np.array(line, np.int32)]
        mask = np.zeros(shape=(image_side, image_side), dtype=np.uint8)
        cv2.polylines(mask, poly, isClosed=False, color=255, thickness=1)
        horizontal_line_masks.append(mask)

    return horizontal_line_masks, vertical_line_masks


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
            digit_img[0, :] = 0
            digit_img[:, 0] = 0
            digit_img[recognize_side - 1, :] = 0
            digit_img[:, recognize_side - 1] = 0

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
