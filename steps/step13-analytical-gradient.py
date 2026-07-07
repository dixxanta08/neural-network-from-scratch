# STEP 13 : Derive gradients for a single neuron and Implement gradient descent
# Look at the handwritten notes for the derivations and more maths

import numpy as np

from nn.neuron import Neuron
from nn.loss import mse


x1 = 1
x2 = 2
target = 4


w1 = 0.5
w2 = 1.0
bias = 0.0

learning_rate = 0.1


neuron = Neuron([w1, w2], bias)
prediction = neuron.forward([x1, x2])

loss = mse([target], [prediction])

print("Prediction:", prediction)
print("Loss:", loss)

# Analytical Gradients

error = prediction - target

grad_w1 = 2 *(prediction- target) * x1
grad_w2 = 2 * (prediction - target) * x2
grad_b = 2*(prediction - target)

print("Gradient w1:", grad_w1)
print("Gradient w2:", grad_w2)
print("Gradient b :", grad_b)


# Gradient Descent Update


w1 = w1 - learning_rate * grad_w1
w2 = w2 - learning_rate * grad_w2
bias = bias - learning_rate * grad_b

print("Updated w1:", w1)
print("Updated w2:", w2)
print("Updated bias:", bias)

# recomputing with updated weights and biases
neuron = Neuron([w1, w2], bias)
prediction = neuron.forward([x1, x2])

loss = mse([target], [prediction])

print("Updated Prediction:", prediction)
print("Updated Loss:", loss)