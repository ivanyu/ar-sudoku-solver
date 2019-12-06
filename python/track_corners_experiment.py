#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np

from sudoku import Corners, Field
from sudoku.solver import clean_image, cut_out_field, binarize_field, detect_grid_points, find_digit_bounding_boxes, \
    enforce_grid_detected


def create_flat_overlay(field: Field, corners: Corners):
    overlay = np.full(shape=(field.image.shape[0], field.image.shape[1], 4), fill_value=(255, 255, 255, 0), dtype=np.uint8)

    bin_field = binarize_field(field)

    grid = detect_grid_points(bin_field)
    enforce_grid_detected(bin_field, grid)

    number_bounding_boxes = find_digit_bounding_boxes(bin_field)
    cell_side = bin_field.side // 9
    for bb in number_bounding_boxes:
        x_cell = (bb.x - bin_field.margin) // cell_side
        y_cell = (bb.y - bin_field.margin) // cell_side

        # number = bin_field.image[bb.y:bb.y + bb.h, bb.x:bb.x + bb.w]
        # print(number.shape)

        cv2.rectangle(overlay, (bb.x, bb.y), (bb.x + bb.w, bb.y + bb.h), (255, 0, 255, 255), lineType=cv2.LINE_AA)
        # cv2.putText(overlay, f"{x_cell}-{y_cell}",
        #             org=(bb.x, bb.y),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5,
        #             color=(0, 255, 0, 255),
        #             lineType=cv2.LINE_AA)

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
prev_corners: Corners = None
prev_flat_overlay = None
while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if ret:
        scale = 0.4
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('frame', frame)

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
            # print(prev_corners.top_left[0] - corners.top_left[0], prev_corners.top_left[1] - corners.top_left[1])
            # print(prev_corners.top_right[0] - corners.top_right[0], prev_corners.top_right[1] - corners.top_right[1])
            # print()

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

            # corners_without_offset = Corners(
            #     (corners.top_left[0], corners.top_left[1] - overlay_min_y),
            #     (corners.top_right[0], corners.top_right[1] - overlay_min_y),
            #     (corners.bottom_right[0], corners.bottom_right[1] - overlay_min_y),
            #     (corners.bottom_left[0], corners.bottom_left[1] - overlay_min_y)
            # )

            # overlay = transform_overlay(
            #     prev_flat_overlay,
            #     (mask_max_y - mask_min_y, image.shape[1]),
            #     Corners((0 + 10, 0 + 10), (320 - 10, 0 + 10), (320 - 10, 320 - 10), (0 + 10, 320 - 10)),
            #     # corners
            #     corners_without_offset
            # )

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

        cv2.imshow("out", out)
        # cv2.imshow("overlay", overlay)
        # cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if ret:
        frame_time_ms = (time.time() - frame_start) * 1000
        print(f"Time per frame: {frame_time_ms}, FPS: {1000 // frame_time_ms}")

cap.release()
cv2.destroyAllWindows()
