# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np


def refine_point(image: np.ndarray, x: int, y: int, win_h_half: int, max_lookaround: int) -> Optional[int]:
    expand_up = 0
    for _ in range(max_lookaround):
        idx = y - win_h_half - 1 - expand_up
        if idx < 0 or image[idx, x] == 0:
            break
        expand_up += 1
    expand_down = 0
    for _ in range(max_lookaround):
        idx = y + win_h_half + expand_down
        if idx >= image.shape[0] or image[idx, x] == 0:
            break
        expand_down += 1

    ys_centered = np.arange(win_h_half * 2 + 1 + expand_up + expand_down) - win_h_half - expand_up
    ys_centered = ys_centered[:, np.newaxis]

    window = image[y - win_h_half - expand_up:y + win_h_half + 1 + expand_down, x:x + 1]
    s = np.sum(window)
    if s > 0:
        avg = np.sum(ys_centered * window) / s
        avg = int(round(avg))
        return y + avg
    else:
        return None


def refine_point_float(image: np.ndarray, x: int, y: int, win_h_half: int, max_lookaround: int) -> Optional[float]:
    expand_up = 0
    for _ in range(max_lookaround):
        idx = y - win_h_half - 1 - expand_up
        if idx < 0 or image[idx, x] == 0:
            break
        expand_up += 1
    expand_down = 0
    for _ in range(max_lookaround):
        idx = y + win_h_half + expand_down
        if idx >= image.shape[0] or image[idx, x] == 0:
            break
        expand_down += 1

    ys_centered = np.arange(win_h_half * 2 + 1 + expand_up + expand_down) - win_h_half - expand_up
    ys_centered = ys_centered[:, np.newaxis]

    window = image[y - win_h_half - expand_up:y + win_h_half + 1 + expand_down, x:x + 1]
    s = np.sum(window)
    if s > 0:
        avg = np.sum(ys_centered * window) / s
        return y + avg
    else:
        return None
