#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import get_line_coeffs


class BorderMapper:
    def __init__(self, border_points):
        """
        Assumes the border points are ordered by the 'from' coordinate.
        """
        self.border_points = border_points

    def map_x(self, y: int) -> int:
        return self._map(y, is_map_x=True)

    def map_y(self, x: int) -> int:
        return self._map(x, is_map_x=False)

    def _map(self, input, is_map_x: bool) -> int:
        if is_map_x:
            input_dim = 1
            output_dim = 0
        else:
            input_dim = 0
            output_dim = 1

        if input < self.border_points[0][input_dim]:
            return self.border_points[0][output_dim]
        if input > self.border_points[-1][input_dim]:
            return self.border_points[1][output_dim]

        less_than = None
        greater_than = None
        # TODO bin search
        for p in self.border_points:
            if p[input_dim] == input:
                return p[output_dim]
            if p[input_dim] < input:
                less_than = p
            elif p[input_dim] > input and greater_than is None:
                greater_than = p
                # No need to iterate further when the first point greater is found.
                break

        assert less_than[input_dim] < input < greater_than[input_dim]

        x1, y1 = less_than
        x2, y2 = greater_than

        if x1 == x2 and is_map_x:
            return x1
        if y1 == y2 and not is_map_x:
            return y1

        a, b = get_line_coeffs(x1, y1, x2, y2)

        if is_map_x:
            x = (input - b) / a
            return int(round(x))
        else:
            y = a * input + b
            return int(round(y))

