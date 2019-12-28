#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sudoku.solver import cut_out_field

_GRID_LINES = 10

_VIZUALIZE = True

if _VIZUALIZE:
    from utils import show_image, wait_windows


def recognize_field(image):
    # Extract the field, its contour and corners.
    field, field_contour, field_corners, perspective_transform_matrix = cut_out_field(image)
    pass
