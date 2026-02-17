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
    v = []
    start = 0

    for point in points:
        b_0, b_1 = estimate_coef(range(start, point), v[start:point])
        v.append((b_0, b_1))
        start = point

    b_0, b_1 = estimate_coef(range(start, temp.size()), v[start:])
    v.append((b_0, b_1))

    return v
