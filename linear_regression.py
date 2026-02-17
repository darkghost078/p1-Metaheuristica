import numpy as np


def estimate_coef(x, y):
    n = np.size(x)

    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    b_0 = SS_xy / SS_xx
    b_1 = m_y - b_0 * m_x

    return (b_0, b_1)


def estimate_all_coef(temp, points):
    coef = []
    start = 0
    points.append(temp.size())

    for point in points:
        b_0, b_1 = estimate_coef(range(start, point), coef[start:point])
        coef.append((b_0, b_1))
        start = point

    return coef


def estimate_point(coef, i):
    return i * coef[0] + coef[1]


def estimate_all_points(coef, points, temp_size):
    estimated = []
    start = 0
    points.append(temp_size)

    for point, pos in points:
        for i in range(start, point):
            estimated.append(estimate_point(coef[pos - 1], i))
        start = point

    return estimated
