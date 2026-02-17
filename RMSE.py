from sklearn.metrics import mean_squared_error

def RMSE (y_real,y_predict):
    return mean_squared_error(y_real,y_predict)


a = [1,5,8,9,65,80]
y = [2,5,10,11,64,96]

print(f"RMSE: {RMSE(a,y)}")