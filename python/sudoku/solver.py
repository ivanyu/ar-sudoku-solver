# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple
from itertools import product

import cv2
import numpy as np

from sudoku import Corners, Field, BoundingBox
from utils import show_image

MAX_INITIAL_IMAGE_SIZE = 500


def load_image(filename: str) -> np.ndarray:
    image = cv2.imread(filename)
    return image


def resize_image(image: np.ndarray) -> np.ndarray:
    max_dim = max(image.shape)
    if max_dim > MAX_INITIAL_IMAGE_SIZE:
        coeff = float(MAX_INITIAL_IMAGE_SIZE) / max_dim
        image = cv2.resize(image, (int(image.shape[1] * coeff), int(image.shape[0] * coeff)))
    return image


def clean_image(image: np.ndarray) -> np.ndarray:
    # image = cv2.GaussianBlur(image, (3, 3), 0)

    # # Adjust brightness.
    # closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE,
    #                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # div_result = np.float32(image) / closed
    # image = np.uint8(cv2.normalize(div_result, None, 0, 255, cv2.NORM_MINMAX))
    return image


def cut_out_field(image: np.ndarray) -> (Field, np.ndarray, Corners, np.ndarray):
    if len(image.shape) == 2:
        bin_image = image
    else:
        bin_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bin_image = cv2.adaptiveThreshold(
            bin_image, maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=55,
            C=6)
    # show_image("bin_image", bin_image, 600)

    field_contour = find_field_contour(bin_image)
    if field_contour is None or len(field_contour.shape) < 3:
        return None, None

    (top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx) = find_field_corners(field_contour)
    corners = Corners(
        field_contour[top_left_idx],
        field_contour[top_right_idx],
        field_contour[bottom_right_idx],
        field_contour[bottom_left_idx],
    )
    # show_corners(bin_image, corners)

    mask = np.zeros(bin_image.shape, np.uint8)
    cv2.drawContours(mask, [field_contour], 0, 255, -1)
    # show_image("mask", mask, 400)

    masked_field = cv2.bitwise_and(image, image, mask=mask)

    mask = cv2.bitwise_not(mask, mask)
    border = cv2.bitwise_or(masked_field, cv2.merge([mask, mask, mask]), mask=mask)
    masked_field = cv2.bitwise_or(masked_field, border)

    # field_side = find_longest_edge_len(corners)
    field_side = 300
    margin = 10
    field, perspective_transform_matrix = warp_field(masked_field, corners, field_side, margin)
    # show_image("field", field)

    return Field(field, field_side, margin), field_contour, corners, perspective_transform_matrix


# def find_longest_edge_len(corners: Corners) -> int:
#     def line_len_sq(a, b) -> int:
#         return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
#
#     longest_edge_len = line_len_sq(corners.top_left, corners.top_right)
#
#     t = line_len_sq(corners.top_right, corners.bottom_right)
#     longest_edge_len = max(longest_edge_len, t)
#
#     t = line_len_sq(corners.bottom_right, corners.bottom_left)
#     longest_edge_len = max(longest_edge_len, t)
#
#     t = line_len_sq(corners.bottom_left, corners.top_left)
#     longest_edge_len = max(longest_edge_len, t)
#
#     return int(sqrt(longest_edge_len))


def find_field_contour(bin_image: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_area_contour: np.ndarray = max(contours, key=cv2.contourArea)
    return largest_area_contour


def find_field_corners(field_contour: np.ndarray) -> (int, int, int, int):
    coord_sums = field_contour.sum(axis=2)
    top_left_idx = np.argmin(coord_sums)
    bottom_right_idx = np.argmax(coord_sums)

    # Diff between y and x. The top-right will have the minimum difference,
    # the bottom-left, the maximum.
    diff = np.diff(field_contour, axis=2)
    top_right_idx = np.argmin(diff)
    bottom_left_idx = np.argmax(diff)

    return top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx


def show_corners(bin_image, corners: Corners):
    color_image = cv2.cvtColor(bin_image, cv2.COLOR_GRAY2BGR)
    radius = max(bin_image.shape[0] // 200, 2)
    color = (0, 0, 255)
    cv2.circle(color_image, tuple(corners.top_left), radius, color, -1)
    cv2.circle(color_image, tuple(corners.top_right), radius, color, -1)
    cv2.circle(color_image, tuple(corners.bottom_right), radius, color, -1)
    cv2.circle(color_image, tuple(corners.bottom_left), radius, color, -1)
    show_image("corners", color_image)


def warp_field(image: np.ndarray, corners: Corners, field_side: int, margin: int) -> Tuple[np.ndarray, np.ndarray]:
    source = np.array([
        corners.top_left,
        corners.top_right,
        corners.bottom_right,
        corners.bottom_left
    ], dtype=np.float32)
    dest = np.array([
        [0 + margin, 0 + margin],
        [field_side + margin, 0 + margin],
        [field_side + margin, field_side + margin],
        [0 + margin, field_side + margin]], dtype=np.float32)
    perspective_transform_matrix = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(
        image,
        perspective_transform_matrix,
        (field_side + 2 * margin, field_side + 2 * margin),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return warped, perspective_transform_matrix


def binarize_field(field: Field) -> Field:
    bin_image = cv2.cvtColor(field.image, cv2.COLOR_BGR2GRAY)
    bin_image = cv2.GaussianBlur(bin_image, (5, 5), 0)
    bin_image = cv2.adaptiveThreshold(
        bin_image, maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=7,
        C=2)
    return Field(bin_image, field.side, field.margin)


def enforce_grid_simple(bin_field: Field):
    """
    Enforces the grid.
    Does modifications in place.
    """
    cell_side = bin_field.side // 9
    lines = cv2.HoughLinesP(
        bin_field.image,
        rho=1,
        theta=np.pi / 180 * 1,  # Pi rad == 180 deg
        threshold=cell_side,
        minLineLength=cell_side * 2,
        maxLineGap=5
    )
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(bin_field.image, (x1, y1), (x2, y2), 255, 1)


def enforce_grid_detected(bin_field: Field, grid):
    for i_row in range(9):
        for i_col in range(9):
            points = grid[i_row:i_row+2, i_col:i_col+2, :].reshape(-1, 2)
            points[[2, 3]] = points[[3, 2]]
            cv2.polylines(bin_field.image, [points], isClosed=True, color=255, thickness=1)


def find_digit_bounding_boxes(field: Field) -> List[BoundingBox]:
    cell_side = field.side // 9
    result = []
    contours, _ = cv2.findContours(field.image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out too big and too small.
        if h > cell_side * 0.9 or w > cell_side * 0.9:
            continue
        if w < cell_side * 0.2:
            continue
        if h < cell_side * 0.5:
            continue
        # Filter out those which out of margin.
        if x <= field.margin or x + w >= field.margin + field.side:
            continue
        result.append(BoundingBox(x, y, w, h))
    return result


def assign_digit_bounding_boxes_to_cells(field: Field, bounding_boxes: List[BoundingBox]) -> List[BoundingBox]:
    cell_side = field.side // 9
    result = [None] * 9 * 9
    for bb in bounding_boxes:
        cell_col = (bb.x - field.margin) // cell_side
        cell_row = (bb.y - field.margin) // cell_side
        assert result[cell_row * 9 + cell_col] is None
        result[cell_row * 9 + cell_col] = bb
    return result


def draw_overlay(image: np.ndarray,
                 overlay: np.ndarray, corners: Corners,
                 field_side: int, margin: int):
    source = np.array([
        corners.top_left,
        corners.top_right,
        corners.bottom_right,
        corners.bottom_left
    ], dtype="float32")
    dest = np.array([
        [0 + margin, 0 + margin],
        [field_side + margin, 0 + margin],
        [field_side + margin, field_side + margin],
        [0 + margin, field_side + margin]], dtype="float32")
    perspective_transform = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(
        overlay,
        perspective_transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.WARP_INVERSE_MAP
    )

    bin_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(bin_warped, 127, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_or(image, warped, mask)
    return result


def detect_grid_points(bin_field: Field):
    assert bin_field.image.shape[0] == bin_field.image.shape[1]
    pieces_n = 3
    piece_side = bin_field.image.shape[0] // pieces_n
    h_x1, h_x2, h_y1, h_y2, v_x1, v_x2, v_y1, v_y2 = collect_line_segments(
        bin_field.image, pieces_n, piece_side
    )
    xs, ys = find_intersections(h_x1, h_x2, h_y1, h_y2, v_x1, v_x2, v_y1, v_y2)
    grid = approximate_grid(bin_field, xs, ys).astype(dtype=int)
    return grid


def collect_line_segments(image, pieces_n, piece_side):
    h_x1 = []
    h_x2 = []
    h_y1 = []
    h_y2 = []
    v_x1 = []
    v_x2 = []
    v_y1 = []
    v_y2 = []
    for piece_y in range(pieces_n):
        offset_y = piece_side * piece_y
        for piece_x in range(pieces_n):
            offset_x = piece_side * piece_x

            piece_h_lines, piece_v_lines = collect_lines_in_grid_piece(
                image,
                piece_side,
                piece_x, piece_y)

            x1, y1, x2, y2 = lines_to_segments(piece_h_lines, piece_side, offset_x, offset_y)
            h_x1.append(x1)
            h_x2.append(x2)
            h_y1.append(y1)
            h_y2.append(y2)

            x1, y1, x2, y2 = lines_to_segments(piece_v_lines, piece_side, offset_x, offset_y)
            v_x1.append(x1)
            v_x2.append(x2)
            v_y1.append(y1)
            v_y2.append(y2)

    h_x1 = np.hstack(h_x1)
    h_x2 = np.hstack(h_x2)
    h_y1 = np.hstack(h_y1)
    h_y2 = np.hstack(h_y2)

    v_x1 = np.hstack(v_x1)
    v_x2 = np.hstack(v_x2)
    v_y1 = np.hstack(v_y1)
    v_y2 = np.hstack(v_y2)

    return h_x1, h_x2, h_y1, h_y2, v_x1, v_x2, v_y1, v_y2


def collect_lines_in_grid_piece(
        field_image,
        piece_side,
        piece_x, piece_y,
):
    offset_x = piece_side * piece_x
    offset_y = piece_side * piece_y

    sub_image = field_image[offset_y:offset_y + piece_side, offset_x:offset_x + piece_side]
    lines = cv2.HoughLines(
        sub_image,
        rho=1,
        theta=np.pi / 180 * 5,  # Pi rad == 180 deg
        threshold=int(piece_side * 0.6)
    )
    if lines is None:
        return
    lines = lines.squeeze()

    # Make rho always positive by rotating the line by Pi rad.
    # If rho is negative, inverse it and subtract Pi from theta.
    # Otherwise, leave as as (np.clip).
    rhos = lines[:, 0:1]
    rho_signs = np.sign(rhos)

    trues = np.ones((lines.shape[0], 1), dtype=bool)
    falses = np.zeros((lines.shape[0], 1), dtype=bool)

    np.multiply(lines, rho_signs, out=lines, where=np.hstack([trues, falses]))

    theta_corrections = np.clip(rho_signs, -1, 0) * np.pi
    np.add(lines, theta_corrections, out=lines, where=np.hstack([falses, trues]))

    lines_struct = np.core.records.fromarrays(lines.transpose(), names='rho, theta', formats='float, float')

    theta_delta = 0.1

    v_line_cond = np.abs(lines_struct['theta']) < theta_delta
    v_lines = lines_struct[v_line_cond]
    v_lines.sort(order=['rho'])
    h_line_cond = np.abs(lines_struct['theta'] - np.pi / 2) < theta_delta
    h_lines = lines_struct[h_line_cond]
    h_lines.sort(order=['rho'])

    # TODO: potential optimization - cluster lines
    # ---

    return h_lines, v_lines


def lines_to_segments(lines, piece_side, offset_x, offset_y):
    # Fill in a and b.
    a = np.cos(lines['theta'])
    b = np.sin(lines['theta'])
    # Fill in x0 and y0.
    x0 = a * lines['rho']
    y0 = b * lines['rho']

    x1 = np.zeros(shape=(lines.shape[0]), dtype=float)
    y1 = np.zeros(shape=(lines.shape[0]), dtype=float)
    x2 = np.zeros(shape=(lines.shape[0]), dtype=float)
    y2 = np.zeros(shape=(lines.shape[0]), dtype=float)
    p = np.zeros(shape=(lines.shape[0]), dtype=float)

    # Strictly vertical lines: b == 0.
    cond = b == 0
    # x1 = x2 = x0
    x1[cond] = x0[cond]
    x2[cond] = x0[cond]
    # y1 = 0
    y1[cond] = 0
    # y2 = piece_side
    y2[cond] = piece_side

    # General lines.
    cond = np.invert(cond)

    # Find lines' intersections with the piece borders.

    def find_intersections_with_borders(x_vec, y_vec, cond, default_x_value):
        # x = default_x_value
        x_vec[cond] = default_x_value
        # p = (x - x0) / (-b)
        np.subtract(x_vec, x0, out=p, where=cond)
        np.divide(p, -b, out=p, where=cond)
        # y = y0 + p * a
        np.multiply(p, a, out=y_vec, where=cond)
        np.add(y_vec, y0, out=y_vec, where=cond)

        # If y is out of borders - clip it to the border and find the corresponding x

        # One pixel out of margin is OK. int() will drop the fraction.
        # On the other hand, without this due to rounding later a line might turn into a point
        # (ex: rho = 0, theta=1.570796251296997)
        cond_y_out_of_borders = np.logical_and(cond, np.logical_or(y_vec < -1.0, y_vec > piece_side + 1.0))

        np.clip(y_vec, 0, piece_side, out=y_vec)
        # p = (y - y0) / a
        np.subtract(y_vec, y0, out=p, where=cond_y_out_of_borders)
        np.divide(p, a, out=p, where=cond_y_out_of_borders)
        # x = x0 + p * (-b)
        np.multiply(p, -b, out=x_vec, where=cond_y_out_of_borders)
        np.add(x_vec, x0, out=x_vec, where=cond_y_out_of_borders)

    # The first point.
    find_intersections_with_borders(x_vec=x1, y_vec=y1, cond=cond, default_x_value=0)
    # The second point.
    find_intersections_with_borders(x_vec=x2, y_vec=y2, cond=cond, default_x_value=piece_side)

    x1 += offset_x
    y1 += offset_y
    x2 += offset_x
    y2 += offset_y

    return x1, y1, x2, y2


def find_intersections(h_x1, h_x2, h_y1, h_y2, v_x1, v_x2, v_y1, v_y2):
    # Calculate line coefficients from cross-product of segment points.
    # See https://stackoverflow.com/a/42727584
    ones = np.ones(shape=(h_x1.shape[0],))
    point1 = np.vstack([h_x1, h_y1, ones])
    point2 = np.vstack([h_x2, h_y2, ones])
    h_lines = np.cross(point1, point2, axis=0)
    h_lines = h_lines.T

    ones = np.ones(shape=(v_x1.shape[0],))
    point1 = np.vstack([v_x1, v_y1, ones])
    point2 = np.vstack([v_x2, v_y2, ones])
    v_lines = np.cross(point1, point2, axis=0)
    v_lines = v_lines.T

    assert (h_lines.shape[0] ==
            h_x1.shape[0] ==
            h_x2.shape[0] ==
            h_y1.shape[0] ==
            h_y2.shape[0])
    assert (v_lines.shape[0] ==
            v_x1.shape[0] ==
            v_x2.shape[0] ==
            v_y1.shape[0] ==
            v_y2.shape[0])

    # Find line intersections from cross-product of line coefficients.
    # See https://stackoverflow.com/a/42727584
    cmb = np.array(list(product(h_lines, v_lines)))
    cr = np.cross(cmb[:, 0, :], cmb[:, 1, :])

    zs = cr[:, 2]
    xs = (cr[:, 0] / zs).reshape(h_lines.shape[0], v_lines.shape[0])
    ys = (cr[:, 1] / zs).reshape(h_lines.shape[0], v_lines.shape[0])

    # Filter out intersection points that don't belong to real segments.
    cond = np.ones(shape=(xs.shape[0], xs.shape[1]), dtype=bool)

    all_v_x = np.vstack([v_x1, v_x2])
    np.logical_and(cond, xs <= np.max(all_v_x, axis=0), out=cond)
    np.logical_and(cond, xs >= np.min(all_v_x, axis=0), out=cond)

    all_v_y = np.vstack([v_y1, v_y2])
    np.logical_and(cond, ys <= np.max(all_v_y, axis=0), out=cond)
    np.logical_and(cond, ys >= np.min(all_v_y, axis=0), out=cond)

    xs = xs.T
    ys = ys.T
    cond = cond.T

    all_h_x = np.vstack([h_x1, h_x2])
    np.logical_and(cond, xs <= np.max(all_h_x, axis=0), out=cond)
    np.logical_and(cond, xs >= np.min(all_h_x, axis=0), out=cond)

    all_h_y = np.vstack([h_y1, h_y2])
    np.logical_and(cond, ys <= np.max(all_h_y, axis=0), out=cond)
    np.logical_and(cond, ys >= np.min(all_h_y, axis=0), out=cond)

    return xs[cond], ys[cond]


def approximate_grid(field, xs, ys):
    grid_rows_cols = 10
    result = np.zeros(shape=(grid_rows_cols, grid_rows_cols, 2), dtype=float)
    cell_side = field.side // 9
    for ideal_row in range(grid_rows_cols):
        for ideal_col in range(grid_rows_cols):
            ideal_x = field.margin + ideal_col * cell_side
            ideal_y = field.margin + ideal_row * cell_side

            t1 = np.subtract(xs, ideal_x)
            np.multiply(t1, t1, out=t1)
            t2 = np.subtract(ys, ideal_y)
            np.multiply(t2, t2, out=t2)
            np.add(t1, t2, out=t1)
            dists = t1
            np.clip(dists, a_min=1, a_max=None, out=dists)

            np.multiply(dists, dists, out=dists)
            # np.multiply(dists, dists, out=dists)
            np.divide(1, dists, out=dists)

            # m_x = np.sum(xs * dists) / np.sum(dists)
            # m_y = np.sum(ys * dists) / np.sum(dists)
            m_x = np.average(xs, weights=dists)
            m_y = np.average(ys, weights=dists)
            result[ideal_row, ideal_col, :] = [m_x, m_y]
    return result
