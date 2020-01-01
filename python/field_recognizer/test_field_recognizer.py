# -*- coding: utf-8 -*-
import unittest
import numpy as np

from sudoku.solver import load_image
from utils import scale_image
from . import recognize_field


class TestFieldRecognizer(unittest.TestCase):
    def test_recognition(self):
        cases = {
            "../images/big-numbers.jpg": np.array([
                [0, 0, 0, 0, 0, 7, 5, 0, 0],
                [7, 0, 0, 1, 0, 0, 0, 4, 0],
                [5, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 1, 3, 9, 0, 0, 0, 8],
                [3, 0, 0, 7, 8, 6, 0, 0, 4],
                [8, 0, 0, 0, 4, 1, 7, 0, 0],
                [0, 0, 8, 0, 0, 0, 0, 0, 9],
                [0, 5, 0, 0, 0, 3, 0, 0, 1],
                [0, 0, 4, 6, 0, 0, 0, 0, 0],
            ])
        }
        cases["../images/warped.jpg"] = cases["../images/big-numbers.jpg"]
        cases["../images/slightly_blurry.jpg"] = cases["../images/big-numbers.jpg"]

        cases["../images/sudoku.jpg"] = np.array([
                [8, 0, 0, 0, 1, 0, 0, 0, 9],
                [0, 5, 0, 8, 0, 7, 0, 1, 0],
                [0, 0, 4, 0, 9, 0, 7, 0, 0],
                [0, 6, 0, 7, 0, 1, 0, 2, 0],
                [5, 0, 8, 0, 6, 0, 1, 0, 7],
                [0, 1, 0, 5, 0, 2, 0, 9, 0],
                [0, 0, 7, 0, 4, 0, 6, 0, 0],
                [0, 8, 0, 3, 0, 9, 0, 4, 0],
                [3, 0, 0, 0, 5, 0, 0, 0, 8],
            ])
        cases["../images/sudoku-rotated.jpg"] = cases["../images/sudoku.jpg"]

        cases["../images/sudoku-1.jpg"] = np.array([
                [0, 0, 0, 6, 0, 4, 7, 0, 0],
                [7, 0, 6, 0, 0, 0, 0, 0, 9],
                [0, 0, 0, 0, 0, 5, 0, 8, 0],
                [0, 7, 0, 0, 2, 0, 0, 9, 3],
                [8, 0, 0, 0, 0, 0, 0, 0, 5],
                [4, 3, 0, 0, 1, 0, 0, 7, 0],
                [0, 5, 0, 2, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 2, 0, 8],
                [0, 0, 2, 3, 0, 1, 0, 0, 0],
            ])

        cases["../images/sudoku-2.jpg"] = np.array([
                [0, 3, 9, 1, 0, 0, 0, 0, 0],
                [4, 0, 8, 0, 6, 0, 0, 0, 2],
                [2, 0, 0, 5, 8, 0, 7, 0, 0],
                [8, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 9, 0, 0, 0],
                [3, 0, 6, 0, 0, 0, 0, 4, 9],
                [0, 0, 0, 0, 1, 0, 0, 3, 0],
                [0, 4, 0, 3, 0, 0, 0, 0, 8],
                [7, 0, 0, 0, 0, 0, 4, 0, 0],
            ])
        cases["../images/sudoku-2-rotated.jpg"] = cases["../images/sudoku-2.jpg"]

        for images_path in cases:
            # if not images_path.endswith("sudoku.jpg"):
            #     continue
            with self.subTest(images_path=images_path):
                expected_field = cases[images_path]
                image = load_image(images_path)
                image = scale_image(image, 640)
                recognized_field = recognize_field(image)
                print(images_path)
                print(expected_field)
                print()
                print(recognized_field)
                print()
                self.assertTrue(np.array_equal(expected_field, recognized_field))
