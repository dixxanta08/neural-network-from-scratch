import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def forward(self, inputs):
        inputs = np.array(inputs)
        return np.dot(inputs, self.weights) + self.bias

