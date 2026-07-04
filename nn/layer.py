import numpy as np

class Layer:

    def __init__(self, neurons, activation=None):
        self.neurons = neurons
        self.activation = activation

    def forward(self, inputs):
        outputs = []

        weight_matrix = np.stack(([neuron.weights for neuron in self.neurons]), axis=1)
        bias_matrix = np.array([neuron.bias for neuron in self.neurons])
        
        outputs = np.matmul(inputs,weight_matrix)+bias_matrix

        if self.activation:
            outputs = [self.activation(output) for output in outputs]
        return outputs
