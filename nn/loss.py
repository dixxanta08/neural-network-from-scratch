import numpy as np

def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(( y_pred- y_true ) ** 2)

def mse_derivate(y_true,y_pred):
    return 2*(y_pred - y_true)/len(y_pred)