
import numpy as np

def relu(x):
    return np.maximum(0, x)

def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
