#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np

from refine_point import refine_point


class TestRefinePoint(unittest.TestCase):
    def test_empty(self):
        img = np.array([
            [0],
            [0],
            [0],
        ])
        r = refine_point(img, 0, 1, 1, 0)
        self.assertIsNone(r)

    def test_center(self):
        img = np.array([
            [0],
            [255],
            [0],
        ])
        r = refine_point(img, 0, 1, 1, 0)
        self.assertEqual(r, 1)

    def test_all(self):
        img = np.array([
            [255],
            [255],
            [255],
        ])
        r = refine_point(img, 0, 1, 1, 0)
        self.assertEqual(r, 1)

    def test_top(self):
        img = np.array([
            [255],
            [0],
            [0],
        ])
        r = refine_point(img, 0, 1, 1, 0)
        self.assertEqual(r, 0)

    def test_top_center(self):
        img = np.array([
            [255],
            [255],
            [0],
        ])
        r = refine_point(img, 0, 1, 1, 0)
        self.assertEqual(r, 1)

    def test_bottom(self):
        img = np.array([
            [0],
            [0],
            [255],
        ])
        r = refine_point(img, 0, 1, 1, 0)
        self.assertEqual(r, 2)

    def test_bottom_center(self):
        img = np.array([
            [0],
            [255],
            [255],
        ])
        r = refine_point(img, 0, 1, 1, 0)
        self.assertEqual(r, 1)

    def test_window_height(self):
        img = np.array([
            [255],
            [0],
            [0],
            [0],
            [0],
        ])
        r = refine_point(img, 0, 1, 1, 0)
        self.assertEqual(r, 0)

    def test_window_look_up(self):
        img = np.array([
            [0],
            [255],
            [255],  # |
            [255],  # |
            [0],    # |
        ])
        r = refine_point(img, 0, 3, 1, 2)
        self.assertEqual(r, 2)

    def test_window_look_down(self):
        img = np.array([
            [0],    # |
            [255],  # |
            [255],  # |
            [255],
            [0],
        ])
        r = refine_point(img, 0, 1, 1, 2)
        self.assertEqual(r, 2)


if __name__ == '__main__':
    unittest.main()
