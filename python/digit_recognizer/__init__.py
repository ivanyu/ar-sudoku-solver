import torch

from .model import DigitRecognizer

_digit_recognizer = DigitRecognizer()
_digit_recognizer.load_state_dict(torch.load("model-ft.pth"))
_digit_recognizer.eval()


def recognize_digits(inputs):
    tensor_inputs = torch.from_numpy(inputs).unsqueeze(1).float()
    outputs = _digit_recognizer(tensor_inputs).data
    result = torch.max(outputs, dim=1)
    # +1 because 0 is excluded from the model
    result = result[1] + 1
    result = list(result.numpy())
    return result
