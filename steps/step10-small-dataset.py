# STEP 09 : STEP 09 : Create a NeuralNetwork class with multiple layers and Implement forward propagation 

import numpy as np 


from nn.neuron import Neuron
from nn.layer import Layer
from nn.activations import step
from nn.neural_network import NeuralNetwork


inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
# and_output = np.array([
#     [0],
#     [0],
#     [0],
#     [1]
# ])

# or_output = np.array([
#     [0],
#     [1],
#     [1],
#     [1]
# ])

rng = np.random.default_rng(seed=42)


neurons_layer_and = [Neuron([1,1], -1.5)]
layer_and = Layer(neurons_layer_and, activation=step)

neural_network_and = NeuralNetwork([layer_and])
print(neural_network_and.forward(inputs))



neurons_layer_or = [Neuron([1,1], -0.5)]
layer_or = Layer(neurons_layer_or, activation=step)

neural_network_or = NeuralNetwork([layer_or])
print(neural_network_or.forward(inputs))



