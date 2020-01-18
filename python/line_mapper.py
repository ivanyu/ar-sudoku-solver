# -*- coding: utf-8 -*-
import bisect
from typing import Union, Collection, Tuple

import numpy as np

from utils import get_line_coeffs


class LineMapper:
    def __init__(self, border_points: Union[np.ndarray, Collection[Tuple[int, int]]]):
        """
        Assumes the line points are ordered by the 'from' coordinate.
        """
        if isinstance(border_points, np.ndarray):
            self._points = [
                list(border_points[:, 0]),
                list(border_points[:, 1]),
            ]
        else:
            xs = []
            ys = []
            for x, y in border_points:
                xs.append(x)
                ys.append(y)
            self._points = [xs, ys]

    def map_x(self, y: int) -> int:
        return self._map(y, is_map_x=True)

    def map_y(self, x: int) -> int:
        return self._map(x, is_map_x=False)

    def _map(self, input, is_map_x: bool) -> int:
        if is_map_x:
            input_coord = self._points[1]
            output_coord = self._points[0]
        else:
            input_coord = self._points[0]
            output_coord = self._points[1]

        if input <= input_coord[0]:
            return int(output_coord[0])
        if input >= input_coord[-1]:
            return int(output_coord[-1])

        insertion_place = bisect.bisect_left(input_coord, input)
        if input_coord[insertion_place] == input:
            return int(output_coord[insertion_place])
        less_than = insertion_place - 1
        greater_than = insertion_place

        assert input_coord[less_than] < input < input_coord[greater_than]

        x1 = self._points[0][less_than]
        y1 = self._points[1][less_than]
        x2 = self._points[0][greater_than]
        y2 = self._points[1][greater_than]

        if x1 == x2 and is_map_x:
            return int(x1)
        if y1 == y2 and not is_map_x:
            return int(y1)

        a, b = get_line_coeffs(x1, y1, x2, y2)

        if is_map_x:
            x = (input - b) / a
            return int(round(x))
        else:
            y = a * input + b
            return int(round(y))
