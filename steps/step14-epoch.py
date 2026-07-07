# STEP 14 : Update weights automatically and Train a single neuron until the loss decreases
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

epochs = 5
for epoch in range(0,epochs):
    print(f"Epoch {epoch} of {epochs}:")
    neuron = Neuron([w1, w2], bias)
    prediction = neuron.forward([x1, x2])

    loss = mse([target], [prediction])

    print("Prediction:", prediction)
    print("Loss:",  round(loss,2))

    # Analytical Gradients

    error = prediction - target

    grad_w1 = 2 *(prediction- target) * x1
    grad_w2 = 2 * (prediction - target) * x2
    grad_b = 2*(prediction - target)



    # Gradient Descent Update


    w1 = w1 - learning_rate * grad_w1
    w2 = w2 - learning_rate * grad_w2
    bias = bias - learning_rate * grad_b


