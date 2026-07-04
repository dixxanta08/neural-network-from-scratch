# Step 06: Create a Layer class containing multiple neurons

from nn.neuron import Neuron
from nn.activations import relu

class Layer:

    def __init__(self, neurons, activation=None):
        self.neurons = neurons
        self.activation = activation

    def forward(self, inputs):
        outputs = []


        outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])

        if self.activation:
            outputs = [self.activation(output) for output in outputs]
        return outputs


import numpy as np

inputs = np.array([0.9, 0.3, 0.2])

weights_1 = np.array([0.6, 0.8, 0.3])
bias_1 = 0.5

weights_2 = np.array([0.5, -1.2, 0.4])
bias_2 = -0.2

weights_3 = np.array([-0.1, 0.5, 0.9])
bias_3 = 0.0


neuron_1 = Neuron(weights_1, bias_1)
neuron_2 = Neuron(weights_2, bias_2)
neuron_3 = Neuron(weights_3, bias_3)

neurons = [neuron_1, neuron_2, neuron_3]

layer = Layer(neurons, activation=relu)

print(layer.forward(inputs))

# [np.float64(1.34), np.float64(-0.02999999999999997), np.float64(0.24)]
# after activation
# [np.float64(1.34), np.float64(0.0), np.float64(0.24)]
