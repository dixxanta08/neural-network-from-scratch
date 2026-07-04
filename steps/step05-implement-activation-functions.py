# Step 05 - Implement activation functions (Step, Sigmoid, ReLU)

"""
An activation function is a set of rules for a computer's "brain" (neural network) that decides if a piece of information is important enough to pass to the next layer. It is essentially the "secret sauce" that allows artificial intelligence to understand complex patterns instead of just doing boring math.
It adds non linearity allowing AI to learn real world complexity
"""

import numpy as np
from nn.neuron import Neuron
# ReLU (Rectified Linear Unit) - outputs input as it is for positive but sets to 0 for negative

def relu(x):
    return np.maximum(0,x)

# Step - returns binary value for value above or below a threshold

def step(x):
    return 1 if x >=0 else 0

#Sigmoid - logistic or squashing function that returns value from 0 to 1

def sigmoid(x):
    return (1/(1+ np.exp(-x)))


# Testing

inputs = np.array([0.9, 0.3, 0.2])
weights = np.array([0.6, 0.8, 0.3])
bias = 0.5

neuron = Neuron(weights, bias)
z = neuron.forward(inputs)
a_relu = relu(z)
a_step = step(z)
a_sigmoid = sigmoid(z)

print("Linear output (z):", z)
print("ReLU output:", a_relu)
print("Step output:", a_step)
print("Sigmoid output:", a_sigmoid)