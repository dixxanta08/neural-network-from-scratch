
import numpy as np

def relu(x):
    return np.maximum(0,x)


def step(x):
    return 1 if x >=0 else 0


def sigmoid(x):
    return (1/(1+ np.exp(-x)))
