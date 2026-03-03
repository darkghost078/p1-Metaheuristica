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


def randomSearch(serie, k, n_iter, prev_best=None):
    if prev_best == None:
        best={'rmse': float('inf'), 'points': None}
    else:
        best = prev_best
    startTime = time.time()
    rmses=[]
    print(n_iter)
    for i in range(n_iter):
        points = randomPoints(len(serie), k)

        rmse = mean_rmse(serie, points)
            
        if rmse < best['rmse']:
            best['rmse'] = rmse
            best['points'] = points

        rmses.append(best)

    endTime = time.time()
    i_time=endTime-startTime

    return best, i_time, rmses