# Step 04(a): Create a reusable Neuron class
# A neuron computes: output = dot(inputs, weights) + bias

import numpy as np


class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def forward(self, inputs):
        inputs = np.array(inputs)
        return np.dot(inputs, self.weights) + self.bias


# Testing the neuron

inputs = np.array([0.9, 0.3, 0.2])

weights = np.array([0.6, 0.8, 0.3])

bias = 0.5

# Create neuron instance
neuron = Neuron(weights, bias)

# Forward pass (compute output)
output = neuron.forward(inputs)

print(output)


# NOTE:
# This implementation will later be moved to nn/neuron.py for reuse across all steps