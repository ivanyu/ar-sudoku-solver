#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import get_line_coeffs


class BorderMapper:
    def __init__(self, border_points):
        """
        Assumes the border points are ordered by the 'from' coordinate.
        """
        self.border_points = border_points

    # def map_y(self, x):
    #
    #     if x < self.border_points[0][0]:
    #         return self.border_points[0][1]
    #     if x > self.border_points[-1][0]:
    #         return self.border_points[1][1]
    #
    #     less_than = None
    #     greater_than = None
    #     # TODO bin search
    #     for p in self.border_points:
    #         if p[0] == x:
    #             return p[1]
    #         if p[0] < x:
    #             less_than = p
    #         elif p[0] > x and greater_than is None:
    #             greater_than = p
    #             # No need to iterate further when the first point greater is found.
    #             break
    #
    #     assert less_than[0] < x < greater_than[0]
    #
    #     x1, y1 = less_than
    #     x2, y2 = greater_than
    #     a, b = self._get_line_coeffs(x1, y1, x2, y2)
    #     y = a * x + b
    #     return int(round(y))

    def map_x(self, y):

        if y < self.border_points[0][1]:
            return self.border_points[0][0]
        if y > self.border_points[-1][1]:
            return self.border_points[1][0]

        less_than = None
        greater_than = None
        # TODO bin search
        for p in self.border_points:
            if p[1] == y:
                return p[0]
            if p[1] < y:
                less_than = p
            elif p[1] > y and greater_than is None:
                greater_than = p
                # No need to iterate further when the first point greater is found.
                break

        assert less_than[1] < y < greater_than[1]

        x1, y1 = less_than
        x2, y2 = greater_than

        if x1 == x2:
            return x1

        a, b = get_line_coeffs(x1, y1, x2, y2)
        y = (y - b) / a
        return int(round(y))
