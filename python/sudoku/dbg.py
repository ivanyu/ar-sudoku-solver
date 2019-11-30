import numpy as np

piece_side = 106
# rho = 5.0
# theta = -0.034906625747680664
rho = 0.0
theta = 1.570796251296997

print('Scalar')

a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho

print('a =', a)
print('b =', b)
print('x0 =', x0)
print('y0 =', y0)

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
    if y2 < -1:
        y2 = 0
        p = (y2 - y0) / a
        x2 = x0 + p * (-b)
    elif y2 > piece_side + 1:
        y2 = piece_side
        p = (y2 - y0) / a
        x2 = x0 + p * (-b)
    x2 = int(x2)
    y2 = int(y2)

print('x1 = ', x1)
print('x2 = ', x2)
print('y1 = ', y1)
print('y2 = ', y2)


#
# print('Vector')
#
# lines = np.array([[rho, theta]])
# lines_to_process = np.core.records.fromarrays(lines.transpose(), names='rho, theta', formats='float, float')
# # print(lines_to_process)
#
# # Fill in a and b.
# a = np.cos(lines_to_process['theta'])
# b = np.sin(lines_to_process['theta'])
# # Fill in x0 and y0.
# x0 = a * lines_to_process['rho']
# y0 = b * lines_to_process['rho']
#
# print('a =', a)
# print('b =', b)
# print('x0 =', x0)
# print('y0 =', y0)
#
# x1 = np.zeros(shape=(lines_to_process.shape[0]), dtype=float)
# y1 = np.zeros(shape=(lines_to_process.shape[0]), dtype=float)
# x2 = np.zeros(shape=(lines_to_process.shape[0]), dtype=float)
# y2 = np.zeros(shape=(lines_to_process.shape[0]), dtype=float)
# p = np.zeros(shape=(lines_to_process.shape[0]), dtype=float)
#
# # Strictly vertical lines: b == 0.
# cond = b == 0
# # x1 = x2 = x0
# x1[cond] = x0[cond]
# x2[cond] = x0[cond]
# # y1 = 0
# y1[cond] = 0
# # y2 = piece_side
# y2[cond] = piece_side
#
# # General lines.
# cond = np.invert(cond)
#
#
# # Find lines' intersections with the piece borders.
#
# def calculate_border_intersections(x_vec, y_vec, cond, default_x_value):
#     # x = default_x_value
#     x_vec[cond] = default_x_value
#     # p = (x - x0) / (-b)
#     np.subtract(x_vec, x0, out=p, where=cond)
#     np.divide(p, -b, out=p, where=cond)
#     # y = y0 + p * a
#     np.multiply(p, a, out=y_vec, where=cond)
#     np.add(y_vec, y0, out=y_vec, where=cond)
#
#     # If y is out of borders - clip it to the border and find the corresponding x
#     cond_y_out_of_borders = np.logical_and(cond, np.logical_or(y_vec < 0, y_vec > piece_side))
#     np.clip(y_vec, 0, piece_side, out=y_vec)
#     # p = (y - y0) / a
#     np.subtract(y_vec, y0, out=p, where=cond_y_out_of_borders)
#     np.divide(p, a, out=p, where=cond_y_out_of_borders)
#     # x = x0 + p * (-b)
#     np.multiply(p, -b, out=x_vec, where=cond_y_out_of_borders)
#     np.add(x_vec, x0, out=x_vec, where=cond_y_out_of_borders)
#
#
# # The first point.
# calculate_border_intersections(x_vec=x1, y_vec=y1, cond=cond, default_x_value=0)
# # The second point.
# calculate_border_intersections(x_vec=x2, y_vec=y2, cond=cond, default_x_value=piece_side)
#
# ones = np.ones(shape=(x1.shape[0],))
# point1 = np.vstack([x1, y1, ones])
# point2 = np.vstack([x2, y2, ones])
# line_coeffs = np.cross(point1, point2, axis=0)
#
# print('x1 = ', x1)
# print('x2 = ', x2)
# print('y1 = ', y1)
# print('y2 = ', y2)



