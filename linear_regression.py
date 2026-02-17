import numpy as np


def estimate_coef(x, y):
    x = np.array(x)
    y = np.array(y)
    n = np.size(x)

    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    b_0 = SS_xy / SS_xx
    b_1 = m_y - b_0 * m_x

    return (b_0.item(), b_1.item())


def estimate_all_coef(temp, points):
    coef = []
    start = 0

    pts = points.copy()
    pts.append(len(temp))

    for point in pts:
        b_0, b_1 = estimate_coef(range(start, point), temp[start:point])
        coef.append((b_0, b_1))
        start = point

    return coef


def estimate_point(coef, i):
    return i * coef[0] + coef[1]


def estimate_all_points(coef, points, temp_size):
    estimated = []
    start = 0

    pts = points.copy()
    pts.append(temp_size)

    for pos, point in enumerate(pts):
        for i in range(start, point):
            estimated.append(estimate_point(coef[pos], i))
        start = point

    return estimated


if __name__ == "__main__":
    temp = [4, 3, 2, 1, 7, 3, 1, 2, 5, 7, 1, 3, 7]
    points = [3, 6, 10]

    coef = estimate_all_coef(temp, points)
    print("Coeficients:", coef)

    estimated = estimate_all_points(coef, points, len(temp))
    print("Estimated points:", estimated)
