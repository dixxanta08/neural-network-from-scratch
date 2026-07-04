# STEP 11 : Implement a loss function (Mean Squared Error) and calculate prediction error also adjust weights manually to observe the effect

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
and_output = np.array([
    [0],
    [0],
    [0],
    [1]
])

or_output = np.array([
    [0],
    [1],
    [1],
    [1]
])

rng = np.random.default_rng(seed=42)


neurons_layer_and = [Neuron([1,1], -1.5)]
layer_and = Layer(neurons_layer_and, activation=step)

neural_network_and = NeuralNetwork([layer_and])
and_pred = neural_network_and.forward(inputs)
print("and_pred:", and_pred)



neurons_layer_or = [Neuron([1,1], -0.5)]
layer_or = Layer(neurons_layer_or, activation=step)

neural_network_or = NeuralNetwork([layer_or])
or_pred = neural_network_or.forward(inputs)
print("or_pred:", or_pred)



# MSE
def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

print("AND dataset MSE:", mse(and_output, and_pred))
print("OR dataset MSE:", mse(or_output, or_pred))


def test_weights_and_bias(weights, bias, inputs, expected_output):
    neuron = Neuron(weights, bias)
    layer = Layer([neuron], activation=step)
    network = NeuralNetwork([layer])

    prediction = network.forward(inputs)

    print(f"Weights: {weights}, Bias: {bias}")
    print("Prediction:", prediction)
    print("MSE:", mse(expected_output, prediction))
    print("-" * 40)

print("------- Changing weights and bias -----------")

test_weights_and_bias([0, 0.5], 0.75, inputs, and_output)
test_weights_and_bias([0, 0], -0.75, inputs, and_output)
test_weights_and_bias([1, 0.85], -1.0, inputs, and_output)