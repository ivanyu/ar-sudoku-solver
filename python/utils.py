# -*- coding: utf-8 -*-
from typing import Optional

import cv2
import numpy as np


def scale_image(img, target_height):
    resize_coeff = target_height / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * resize_coeff), int(img.shape[0] * resize_coeff)), interpolation=cv2.INTER_CUBIC)


def show_image(window_name: str, image: np.ndarray, target_height: Optional[int] = None):
    if target_height is None:
        cv2.imshow(window_name, image)
    else:
        cv2.imshow(window_name, scale_image(image, target_height))


def wait_windows():
    while True:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def get_line_coeffs(x1, y1, x2, y2):
    assert x1 != x2
    a = (y2 - y1) / (x2 - x1)
    b = y2 - x2 * a
    return a, b
