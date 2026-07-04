# STEP 08: Process a batch of inputs at once

import numpy as np 
from nn.neuron import Neuron
from nn.layer import Layer
from nn.activations import relu


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

neurons = [neuron_1, neuron_2, neuron_3]

layer = Layer(neurons, activation=relu)



print(layer.forward(inputs))
