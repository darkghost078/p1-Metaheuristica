from sklearn.metrics import mean_squared_error
from linear_regression import estimate_all_coef, estimate_all_points
import numpy as np

def RMSE (y_real,y_predict):
    mse = mean_squared_error(y_real,y_predict)
    return np.sqrt(mse)

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

# if __name__ == "__main__":
#     a = [1,5,8,9,65,80]
#     y = [2,5,10,11,64,96]

# print(f"RMSE: {RMSE(a,y)}")