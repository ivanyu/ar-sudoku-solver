# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torchvision as tv

from .model import DigitRecognizer2


def create_recognizer():
    _digit_recognizer: DigitRecognizer2 = DigitRecognizer2()
    _path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model-ft.pth"
    )
    _digit_recognizer.load_state_dict(torch.load(_path))
    _digit_recognizer.eval()

    def recognize_digits(images):
        inputs = []
        for img in images:
            input = tv.transforms.functional.to_tensor(img)
            input = tv.transforms.functional.normalize(input, (0.5,), (0.5,), inplace=True)
            inputs.append(input)
        inputs = torch.cat(inputs).unsqueeze(1)
        outputs = _digit_recognizer(inputs).data
        result = torch.max(outputs, dim=1)
        # +1 because class 0 is 1 and so on
        result = result[1] + 1
        result = list(result.numpy())
        return result, outputs

    return recognize_digits
