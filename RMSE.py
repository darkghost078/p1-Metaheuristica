from sklearn.metrics import mean_squared_error
import numpy as np

def RMSE (y_real,y_predict):
    mse = mean_squared_error(y_real,y_predict)
    return np.sqrt(mse)

if __name__ == "__main__":
    a = [1,5,8,9,65,80]
    y = [2,5,10,11,64,96]

print(f"RMSE: {RMSE(a,y)}")