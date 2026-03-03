import time
from linear_regression import estimate_all_coef, estimate_all_points
from puntosRandom import randomPoints
from RMSE import RMSE

def mean_rmse(serie, points):
    rmse_acc = 0.0
    start = 0

    coef = estimate_all_coef(serie, points.copy())
    estimated = estimate_all_points(coef, points.copy(), len(serie))

    pts = points.copy()
    pts.append(len(serie))

    for point in pts:
        rmse_acc += RMSE(serie[start:point], estimated[start:point])
        start = point

    return rmse_acc / len(pts)


def randomSearch(serie, k, n_iter):
    best = {'rmse': float('-inf'), 'points': None}
    times = []
    startTime = time.time()
    rmses=[]

    for i in range(n_iter):
        points = randomPoints(len(serie), k)

        rmse = mean_rmse(serie, points)
        endTime = time.time()

        times.append(endTime - startTime)
        
        if rmse > best['rmse']:
            best['rmse'] = rmse
            best['points'] = points

        rmses.append(best)

    return best, times, rmses