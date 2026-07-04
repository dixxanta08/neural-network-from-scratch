import numpy as np

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
