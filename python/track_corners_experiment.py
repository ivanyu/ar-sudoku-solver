#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np

from digit_recognizer import recognize_digits
from sudoku import Corners, Field
from sudoku.solver import clean_image, cut_out_field, binarize_field, detect_grid_points, find_digit_bounding_boxes, \
    enforce_grid_detected, assign_digit_bounding_boxes_to_cells
from video_out import VideoOut, NoOpVideoOut


MS_PER_FRAME_30_FPS = 33


def create_flat_overlay(field: Field, corners: Corners):
    overlay = np.full(shape=(field.image.shape[0], field.image.shape[1], 4),
                      fill_value=(255, 255, 255, 0), dtype=np.uint8)

    bin_field = binarize_field(field)

    grid = detect_grid_points(bin_field)
    enforce_grid_detected(bin_field, grid)

    digit_bounding_boxes = find_digit_bounding_boxes(bin_field)
    digit_bounding_boxes_by_cells = assign_digit_bounding_boxes_to_cells(field, digit_bounding_boxes)

    digits_for_recog = []
    digits_for_recog_coords = []
    for i, bbox in enumerate(digit_bounding_boxes_by_cells):
        if bbox is None:
            continue

        digit = bin_field.image[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]

        i_row = i // 9
        i_col = i % 9

        digit = cv2.morphologyEx(digit, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))

        digit = cv2.copyMakeBorder(digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
        target_h = 28
        target_w = 28
        scale = float(target_h) / digit.shape[0]
        digit = cv2.resize(digit, dsize=(int(digit.shape[1] * scale), target_h))
        w_diff = target_w - digit.shape[1]
        assert w_diff >= 0
        digit = cv2.copyMakeBorder(digit, 0, 0, w_diff // 2, w_diff - (w_diff // 2), cv2.BORDER_CONSTANT)
        digits_for_recog.append(digit[np.newaxis, :, :])
        digits_for_recog_coords.append((i_row, i_col))

    digits_for_recog = np.vstack(digits_for_recog)
    labels = recognize_digits(digits_for_recog)
    for i, (i_row, i_col) in enumerate(digits_for_recog_coords):
        bbox = digit_bounding_boxes_by_cells[i_row * 9 + i_col]
        cv2.rectangle(overlay, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (255, 0, 255, 255), lineType=cv2.LINE_AA)
        cv2.putText(overlay, str(labels[i]),
                    org=(bbox.x + 20, bbox.y + 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 255, 255),
                    lineType=cv2.LINE_AA)
    return overlay


def transform_overlay(old_overlay, new_shape, old_corners: Corners, new_corners: Corners):
    source = np.array([
        old_corners.top_left,
        old_corners.top_right,
        old_corners.bottom_right,
        old_corners.bottom_left
    ], dtype="float32")
    dest = np.array([
        new_corners.top_left,
        new_corners.top_right,
        new_corners.bottom_right,
        new_corners.bottom_left
    ], dtype="float32")
    perspective_transform = cv2.getPerspectiveTransform(source, dest)
    warped_overlay = cv2.warpPerspective(
        old_overlay,
        perspective_transform,
        (new_shape[1], new_shape[0]),
        flags=cv2.INTER_CUBIC
    )
    return warped_overlay


# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("../images/VID_20191201_104129.mp4")
cap = cv2.VideoCapture("../images/VID_20191204_201215.mp4")

# INPUT_SCALE = 0.3
# input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * INPUT_SCALE)
# input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * INPUT_SCALE)
input_width = 640
input_height = 360

prev_corners: Corners = None
prev_flat_overlay = None

# video_out = VideoOut("output")
video_out = NoOpVideoOut()

while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("frame", frame)

        image = clean_image(frame)
        field, corners, perspective_transform_matrix = cut_out_field(image)
        assert field is not None

        overlay_min_y = None
        overlay_max_y = None
        if prev_corners is None:
            prev_corners = corners
            flat_overlay = create_flat_overlay(field, corners)
            prev_flat_overlay = flat_overlay
            overlay = cv2.warpPerspective(
                flat_overlay,
                perspective_transform_matrix,
                (image.shape[1], image.shape[0]),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
            )
            overlay_min_y = 0
            overlay_max_y = image.shape[0]
        else:
            corner_ys = [corners.top_left[1], corners.top_right[1], corners.bottom_left[1], corners.bottom_right[1]]
            overlay_min_y = min(corner_ys)
            overlay_max_y = max(corner_ys)

            source = np.array([
                (0 + 10, 0 + 10),
                (320 - 10, 0 + 10),
                (320 - 10, 320 - 10),
                (0 + 10, 320 - 10)
            ], dtype="float32")
            dest = np.array([
                (corners.top_left[0], corners.top_left[1] - overlay_min_y),
                (corners.top_right[0], corners.top_right[1] - overlay_min_y),
                (corners.bottom_right[0], corners.bottom_right[1] - overlay_min_y),
                (corners.bottom_left[0], corners.bottom_left[1] - overlay_min_y)
            ], dtype="float32")
            perspective_transform = cv2.getPerspectiveTransform(source, dest)
            overlay = cv2.warpPerspective(
                prev_flat_overlay,
                perspective_transform,
                (image.shape[1], overlay_max_y - overlay_min_y),
                flags=cv2.INTER_CUBIC
            )

        out = frame
        overlay, mask = np.split(overlay, indices_or_sections=[3], axis=2)

        mask = mask / 255.0
        affected_out = out[overlay_min_y:overlay_max_y, :, :]
        np.multiply(affected_out, (1 - mask), out=affected_out, casting="unsafe")
        np.multiply(overlay, mask, out=overlay, casting="unsafe")
        np.add(affected_out, overlay, out=affected_out)

        frame_time_ms = (time.time() - frame_start) * 1000
        cv2.putText(out,
                    "FPS: {0:.2f}".format(1000 / frame_time_ms),
                    org=(5, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 100, 255, 255),
                    lineType=cv2.LINE_AA)
        cv2.putText(out,
                    f"{input_width}x{input_height}",
                    org=(5, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 100, 255, 255),
                    lineType=cv2.LINE_AA)

        cv2.imshow("out", out)
        video_out.write_frame(out)

    frame_time_ms = (time.time() - frame_start) * 1000
    frame_wait = max(1, int(MS_PER_FRAME_30_FPS - frame_time_ms))
    if cv2.waitKey(frame_wait) & 0xFF == ord('q'):
        break

    if ret:
        print(f"Time per frame: {frame_time_ms}, FPS: {1000 // frame_time_ms}")

video_out.release()
cap.release()
cv2.destroyAllWindows()
