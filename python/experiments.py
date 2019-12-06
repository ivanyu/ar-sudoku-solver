import cv2
import numpy as np

from sudoku import Field
from sudoku.solver import show_image


def detect_and_draw_grid(bin_field: Field):
    assert bin_field.image.shape[0] == bin_field.image.shape[1]
    pieces_n = 3
    piece_side = bin_field.image.shape[0] // pieces_n
    imgx = cv2.cvtColor(bin_field.image, cv2.COLOR_GRAY2BGR)
    vertical_mask = np.zeros(bin_field.image.shape, np.uint8)
    horizontal_mask = np.zeros(bin_field.image.shape, np.uint8)
    for i in range(pieces_n):
        offset_y = piece_side * i
        for j in range(pieces_n):
            offset_x = piece_side * j
            sub_image = bin_field.image[offset_y:offset_y + piece_side, offset_x:offset_x + piece_side]
            show_image("sub_image1", sub_image)
            lines = cv2.HoughLines(
                sub_image,
                rho=1,
                theta=np.pi / 180,  # Pi rad == 180 deg
                threshold=int(piece_side * 0.6)
            )
            if lines is None:
                continue
            lines = lines.squeeze()

            theta_d = 0.1

            horizontal = []
            vertical = []
            for rho, theta in lines:
                if rho < 0:
                    rho = -rho
                    theta -= np.pi

                if abs(theta) < theta_d:
                    vertical.append((rho, theta))
                elif abs(theta - np.pi / 2) < theta_d:
                    horizontal.append((rho, theta))

            # Sort by rho.
            horizontal = sorted(horizontal, key=lambda x: x[0])
            vertical = sorted(vertical, key=lambda x: x[0])

            rho_cluster_d = 5
            horizontal_new = [horizontal[0]]
            for g in range(1, len(horizontal)):
                if abs(horizontal_new[-1][1] - horizontal[g][1]) > theta_d or abs(horizontal_new[-1][0] - horizontal[g][0]) > rho_cluster_d:
                    horizontal_new.append(horizontal[g])
                else:
                    horizontal_new[-1] = (
                        np.average([horizontal_new[-1][0], horizontal[g][0]]),
                        np.average([horizontal_new[-1][1], horizontal[g][1]])
                    )
            horizontal = horizontal_new

            vertical_new = [vertical[0]]
            for g in range(1, len(vertical)):
                if abs(vertical_new[-1][1] - vertical[g][1]) > theta_d or abs(vertical_new[-1][0] - vertical[g][0]) > rho_cluster_d:
                    vertical_new.append(vertical[g])
                else:
                    vertical_new[-1] = (
                        np.average([vertical_new[-1][0], vertical[g][0]]),
                        np.average([vertical_new[-1][1], vertical[g][1]])
                    )
            vertical = vertical_new

            def draw_line(rho, theta, mask, color):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                if b == 0:
                    x1 = x2 = int(x0)
                    y1 = 0
                    y2 = int(piece_side)
                else:
                    x1 = 0
                    p = (x1 - x0) / (-b)
                    y1 = y0 + p * a
                    if y1 < 0:
                        y1 = 0
                        p = (y1 - y0) / a
                        x1 = x0 + p * (-b)
                    elif y1 > piece_side:
                        y1 = piece_side
                        p = (y1 - y0) / a
                        x1 = x0 + p * (-b)
                    x1 = int(x1)
                    y1 = int(y1)

                    x2 = piece_side
                    p = (x2 - x0) / (-b)
                    y2 = y0 + p * a
                    if y2 < 0:
                        y2 = 0
                        p = (y2 - y0) / a
                        x2 = x0 + p * (-b)
                    elif y2 > piece_side:
                        y2 = piece_side
                        p = (y2 - y0) / a
                        x2 = x0 + p * (-b)
                    x2 = int(x2)
                    y2 = int(y2)

                cv2.line(imgx, (offset_x + x1, offset_y + y1), (offset_x + x2, offset_y + y2), color, 5)
                cv2.line(mask, (offset_x + x1, offset_y + y1), (offset_x + x2, offset_y + y2), 255, 1)

            for rho, theta in horizontal:
                draw_line(rho, theta, horizontal_mask, (0, 0, 0))
            for rho, theta in vertical:
                draw_line(rho, theta, vertical_mask, (0, 0, 0))
    show_image("imgx", imgx)
    show_image("horizontal_mask", horizontal_mask)
    show_image("vertical_mask", vertical_mask)
    intersection = cv2.bitwise_and(horizontal_mask, vertical_mask)
    show_image("intersection", intersection)
