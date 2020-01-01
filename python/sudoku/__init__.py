# -*- coding: utf-8 -*-
from collections import namedtuple
from dataclasses import dataclass

import numpy as np


@dataclass
class Corners:
    top_left: (int, int)
    top_right: (int, int)
    bottom_right: (int, int)
    bottom_left: (int, int)


@dataclass
class Field:
    image: np.ndarray
    side: int
    margin: int

    def ideal_cell_side(self):
        return self.side // 9


BoundingBox = namedtuple('BoundingBox', 'x y w h')

DISPLAY = True
# DISPLAY = False
