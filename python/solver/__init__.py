# -*- coding: utf-8 -*-
from itertools import product
from typing import Optional
from copy import deepcopy

import numpy as np

_FIELD_SIDE = 9
_BOX_SIDE = 3


def solve(field: np.array) -> Optional[np.array]:
    assert field.shape == (_FIELD_SIDE, _FIELD_SIDE)

    variants = np.full(shape=(_FIELD_SIDE, _FIELD_SIDE, _FIELD_SIDE + 1), fill_value=True)

    # Exclude already filled-in digits from the cell's variants.
    m = np.array(list(product(range(_FIELD_SIDE), range(_FIELD_SIDE))))
    variants[m[:, 0], m[:, 1], field[m[:, 0], m[:, 1]]] = False

    variants_in_rows = np.all(variants, axis=1)
    variants_in_rows = np.repeat(variants_in_rows[:, np.newaxis, :], _FIELD_SIDE, axis=1)

    variants_in_cols = np.all(variants, axis=0)
    variants_in_cols = np.repeat(variants_in_cols[np.newaxis, :, :], _FIELD_SIDE, axis=0)

    # Restrict variants in boxes in place.
    for box_row in range(_BOX_SIDE):
        for box_col in range(_BOX_SIDE):
            r = slice(box_row * _BOX_SIDE, (box_row + 1) * _BOX_SIDE)
            c = slice(box_col * _BOX_SIDE, (box_col + 1) * _BOX_SIDE)
            box = variants[r, c]
            variants[r, c] = np.all(np.all(box, axis=1), axis=0)

    np.bitwise_and(variants, variants_in_rows, out=variants)
    np.bitwise_and(variants, variants_in_cols, out=variants)

    return _solve0(field, variants)


def _solve0(field: np.array, variants: np.array) -> Optional[np.array]:
    # If all cells are filled:
    if np.all(field != 0):
        return field

    # Count available variants in cells.
    num_variants = np.sum(variants, axis=2)

    # Placeholder to prevent reprocessing of filled cells.
    num_variants[field != 0] = 1000

    # Find the most restricted cell at the moment and try to fill it.
    most_restricted_cell_idx = np.argmin(num_variants)
    cur_cell = np.unravel_index(most_restricted_cell_idx, num_variants.shape)
    assert field[cur_cell] == 0

    variants_for_cell = variants[cur_cell]

    # If variants for this cell:
    if not np.any(variants_for_cell):
        return None

    for v in np.where(variants_for_cell)[0]:  # where returns tuple
        assert v != 0

        field_next = deepcopy(field)
        variants_next = deepcopy(variants)

        field_next[cur_cell] = v

        # Remove just added value from possible variants in the cell itself.
        variants_next[cur_cell][v] = False

        cur_cell_row, cur_cell_col = cur_cell
        # Remove just added value from possible variants in the row.
        variants_next[cur_cell_row, :, v] = False
        # Remove just added value from possible variants in the col.
        variants_next[:, cur_cell_col, v] = False
        # Remove just added value from possible variants in the box.
        box_row = cur_cell_row // _BOX_SIDE
        box_col = cur_cell_col // _BOX_SIDE
        r = slice(box_row * _BOX_SIDE, (box_row + 1) * _BOX_SIDE)
        c = slice(box_col * _BOX_SIDE, (box_col + 1) * _BOX_SIDE)
        variants_next[r, c, v] = False

        res = _solve0(field_next, variants_next)
        if res is not None:
            return res

    return None
