# STEP 16: again finally done
# STEP 16 : Train on the XOR dataset (observe failure with one layer) and Add a hidden layer and train until XOR is solved


import numpy as np


def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x)/((1+np.exp(-x))**2)
    else:
        return 1 / (1 + np.exp(-x))


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(( y_pred- y_true ) ** 2)

def mse_derivate(y_true,y_pred):
    return 2*(y_pred - y_true)/len(y_pred)


class Neuron:

    def __init__ (self, weights,bias):
        self.weights = weights
        self.bias = bias
        



class Layer:

    def __init__(self,neurons, activation=sigmoid):
        self.neurons = neurons
        self.activation= activation
        self.grad_weights = []
        self.grad_bias = []

    def forward(self,inputs ):
        self.inputs = inputs
        weight_matrix = np.array([neuron.weights for neuron in self.neurons])
        bias_vector = np.array([neuron.bias for neuron in self.neurons])
        z = np.dot(inputs , weight_matrix.T) +  bias_vector 
        self.z = z
        return self.activation(z)
    
    


    
class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers
        self.outputs =[]
        self.loss = []
        self.deltas = [None]*len(layers)
        
    
    def forward(self, inputs):
        layer_input = inputs
        layer_output = 0
        for layer in self.layers:
            layer_output = layer.forward(layer_input)
            layer_input = layer_output
        return layer_output

    def train(self, inputs, targets, learning_rate=0.1, epochs=1):
        for epoch in range(epochs):
            print(f"\n==============================\nEPOCH {epoch} of {epochs}\n==============================")
            result = self.forward(inputs)
            self.outputs.append(result)
            loss = mse(targets, result)
            self.loss.append(loss)

            loss_derivate = mse_derivate(targets, result)
            self.deltas[-1] = loss_derivate * sigmoid(self.layers[-1].z,derivate=True)
            # for hidden layer
            for i in range(len(self.layers) - 2, -1, -1):
                hidden_layer_delta = (self.deltas[i+1] @ np.array([neuron.weights for neuron in self.layers[i+1].neurons])) * sigmoid(self.layers[i].z, derivate=True)
                self.deltas[i] = hidden_layer_delta
            
            # compute gradients
            for index,layer in enumerate(self.layers):
                layer.grad_weights = ( layer.inputs.T @ self.deltas[index])/len(layer.inputs)
                # layer.inputs.T              -> (number of inputs, number of samples)
                # self.deltas[index]          -> (number of samples, number of neurons)
                # layer.inputs.T @ deltas     -> (number of inputs, number of neurons)
                # / len(layer.inputs)         -> same shape (number of inputs, number of neurons)
                # layer.grad_weights          -> (number of inputs, number of neurons)
                layer.grad_bias = np.mean(self.deltas[index],axis=0)
            
            # update weights and bias
            for index,layer in enumerate(self.layers):
                # weights 
                layer_weights = np.array([neuron.weights for neuron in layer.neurons])
                new_layer_weights = layer_weights - learning_rate * layer.grad_weights.T
                for n_index,neuron in enumerate(layer.neurons):
                    neuron.weights = new_layer_weights[n_index]

                #bias
                
                layer_bias = np.array([neuron.bias for neuron in layer.neurons])
                new_layer_bias = layer_bias - learning_rate * layer.grad_bias.T
                
                for n_index,neuron in enumerate(layer.neurons):
                    neuron.bias = new_layer_bias[n_index]


                

    def predict(self, input):
        return self.forward(input)
    

rng = np.random.default_rng(42)

hidden_neurons = [
    Neuron(rng.uniform(-1, 1, 2), rng.uniform(-1, 1))
    for _ in range(2)
]

output_neurons = [
    Neuron(rng.uniform(-1, 1, 2), rng.uniform(-1, 1))
]

hidden_layer = Layer(hidden_neurons, activation=sigmoid)
output_layer = Layer(output_neurons, activation=sigmoid)

nn = NeuralNetwork([hidden_layer, output_layer])

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

targets = np.array([
    [0],
    [1],
    [1],
    [0]
])

nn.train(inputs, targets, learning_rate=0.5, epochs=10000)


print("----- Prediction Results -----")
test_cases = [
    ([1, 1], 0),
    ([0, 0], 0),
    ([1, 0], 1),
    ([0, 1], 1)
]
for inputs, expected in test_cases:
    raw_pred = nn.predict(inputs)
    rounded_pred = int(np.round(raw_pred)[0]) 
    print(f"Input: {inputs} -> Predicted: {rounded_pred} (Expected: {expected})")
