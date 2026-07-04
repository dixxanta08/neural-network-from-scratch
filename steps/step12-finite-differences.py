# STEP 12 : Implement numerical gradient estimation (finite differences)
import numpy as np 


from nn.neuron import Neuron
from nn.layer import Layer
from nn.activations import step
from nn.neural_network import NeuralNetwork
from nn.loss import mse


inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
and_output = np.array([
    [0],
    [0],
    [0],
    [1]
])

rng = np.random.default_rng(seed=42)


def get_neuron_loss(w1,w2,bias):
    neurons = [Neuron([w1,w2],bias)]
    layer = Layer(neurons, activation=step)

    neural_network = NeuralNetwork([layer])
    prediction = neural_network.forward(inputs)

    return mse(and_output, prediction)



base_loss = get_neuron_loss(0.25,0.5,-0.5)
loss_weight_1 = get_neuron_loss(0.5,0.5,-0.5)
loss_weight_2 = get_neuron_loss(0.25,0.75,-0.5)
loss_bias = get_neuron_loss(0.25,0.5,-0.75)

h = 0.25

print(base_loss,loss_weight_1)
print("finite difference (w1): ", (loss_weight_1 - base_loss)/h) # 1.0 means shift w1 opposite direction
print("finite difference (w2): ", (loss_weight_2 - base_loss)/h) # 0 means let base loss as it is
print("finite difference (bias): ", (loss_bias - base_loss)/h) # -1.0 means shift bias same direction




