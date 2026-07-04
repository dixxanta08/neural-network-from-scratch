# STEP 09 : STEP 09 : Create a NeuralNetwork class with multiple layers and Implement forward propagation 

import numpy as np 


from nn.neuron import Neuron
from nn.layer import Layer
from nn.activations import relu

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        layer_input = inputs
        for layer in self.layers:
            layer_input = layer.forward(layer_input)
            print("layer_1 output:",layer_input) 
        output = layer_input
        return output
        



inputs = np.array([
    [0.9, 0.3, 0.2],
    [0.55,0.76,0.34],
    [0.2,0.4,0.2],
    [0.56,0.76,0.9],
])

weights_1 = np.array([0.6, 0.8, 0.3])
bias_1 = 0.5

weights_2 = np.array([0.5, -1.2, 0.4])
bias_2 = -0.2

weights_3 = np.array([-0.1, 0.5, 0.9])
bias_3 = 0.0



neuron_1 = Neuron(weights_1, bias_1)
neuron_2 = Neuron(weights_2, bias_2)
neuron_3 = Neuron(weights_3, bias_3)
rng = np.random.default_rng(seed=42)


neurons_layer_1 = [Neuron(np.array(rng.random(3)), rng.random()) for _ in range(0,8)]
neurons_layer_2 = [Neuron(np.array(rng.random(8)), rng.random()) for _ in range(0,4)]
neurons_layer_3 = [Neuron(np.array(rng.random(4)), rng.random()) for _ in range(0,1)]
layer_1 = Layer(neurons_layer_1, activation=relu)
layer_2 = Layer(neurons_layer_2, activation=relu)
layer_3 = Layer(neurons_layer_3, activation=relu)

neural_network = NeuralNetwork([layer_1,layer_2,layer_3])
print(neural_network.forward(inputs))



