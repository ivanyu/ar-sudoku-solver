import unittest
import numpy as np

from line_mapper import LineMapper


class TestLineMapper(unittest.TestCase):
    _LINE_1 = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
    ])

    _LINE_2 = np.array([
        [0, 0],
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40],
        [50, 50],
        [60, 60],
        [70, 70],
        [80, 80],
        [90, 90],
        [100, 100],
    ])

    def test_far_left_y(self):
        mapper = LineMapper(self._LINE_1)
        self.assertEqual(mapper.map_y(0), 10)

    def test_far_left_x(self):
        mapper = LineMapper(self._LINE_1)
        self.assertEqual(mapper.map_x(0), 1)

    def test_far_right_y(self):
        mapper = LineMapper(self._LINE_1)
        self.assertEqual(mapper.map_y(10), 30)

    def test_far_right_x(self):
        mapper = LineMapper(self._LINE_1)
        self.assertEqual(mapper.map_x(100), 3)

    def test_exact_match(self):
        mapper = LineMapper(self._LINE_2)
        for x, y in self._LINE_2:
            self.assertEqual(mapper.map_y(x), y)
            self.assertEqual(mapper.map_x(y), x)

    def test_interpolate(self):
        mapper = LineMapper(self._LINE_2)
        for c in range(self._LINE_2[0, 0], self._LINE_2[-1, 0] + 1):
            self.assertEqual(mapper.map_x(c), c)
            self.assertEqual(mapper.map_y(c), c)


if __name__ == '__main__':
    unittest.main()
