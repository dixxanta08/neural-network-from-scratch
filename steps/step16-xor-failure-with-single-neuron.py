# STEP 16: XOR dataset
import numpy as np 


class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = 0.0
        

    def forward(self, inputs):
        inputs = np.array(inputs)
        return np.dot(inputs, self.weights) + self.bias

class Layer:

    def __init__(self, neurons, activation=None):
        self.neurons = neurons
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        outputs = []
        weight_matrix = np.stack(([neuron.weights for neuron in self.neurons]), axis=1)
        bias_matrix = np.array([neuron.bias for neuron in self.neurons])        
    
        outputs = np.matmul(inputs,weight_matrix)+bias_matrix
        self.z = outputs

        if self.activation:
            outputs = self.activation(outputs)
        return outputs
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    return np.exp(-x)/((1+np.exp(-x))**2)


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(( y_pred- y_true ) ** 2)

def mse_derivate(y_true,y_pred):
    return 2*(y_pred - y_true)/len(y_pred)


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers
        self.losses = []

    def forward(self, inputs):
        layer_input = inputs
        for layer in self.layers:
            layer_input = layer.forward(layer_input)
        output = layer_input
        return output
    
    def train(self, inputs,epochs, targets, alpha):
        for epoch in range(0, epochs):
            print("=======================================")
            print(f"Epoch {epoch}/{epochs}")
            for i, layer in enumerate(self.layers):
                print(f"Layer {i}")
               
                for j, neuron in enumerate(layer.neurons):
                    print(f"Neuron {j}: Weights = {neuron.weights}, Bias = {neuron.bias}")


            epoch_output = self.forward(inputs)
            epoch_loss = mse(targets, epoch_output)
            self.losses.append(epoch_loss)
            print(f"epoch {epoch}'s output: {epoch_output}")
            print(f"epoch {epoch}'s loss: {epoch_loss}")
            # analytical gradient approach
            
            deltas = [None] * len(self.layers)

            last_layer = self.layers[-1]


            deltas[-1] = (
                mse_derivate(targets, epoch_output) *
                sigmoid_derivate(last_layer.z)
            )


            for layer_index in reversed(range(len(self.layers))):
                layer = self.layers[layer_index]

               
                if layer_index == len(self.layers) - 1:
                    deltas[layer_index] = (
                        mse_derivate(targets, epoch_output) *
                        sigmoid_derivate(layer.z)
                    )
                else:
                    next_layer = self.layers[layer_index + 1]
                    weights_next = np.array([n.weights for n in next_layer.neurons]).T

                    deltas[layer_index] = (
                        deltas[layer_index + 1] @ weights_next
                    ) * sigmoid_derivate(layer.z)

               
                for neuron_index, neuron in enumerate(layer.neurons):

                    delta = deltas[layer_index][:, neuron_index]

                
                    for w_index in range(len(neuron.weights)):
                        gradient = np.mean(delta * layer.inputs[:, w_index])
                        neuron.grad_weights[w_index] = gradient

                    neuron.grad_bias = np.mean(delta)

            for layer_index in range(len(self.layers)):
                layer = self.layers[layer_index]
                for neuron_index in range(len(layer.neurons)):
                    layer.neurons[neuron_index].weights -= layer.neurons[neuron_index].grad_weights * alpha 
                    layer.neurons[neuron_index].bias -= layer.neurons[neuron_index].grad_bias * alpha 
                    

                    

inputs = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1],
])
targets = np.array([
    [0],
    [1],
    [1],
    [0]
])


rng = np.random.default_rng(seed=42)


neurons_layer_1 = [Neuron(np.array(rng.random(2)), rng.random()) for _ in range(0,1)]
layer_1 = Layer(neurons_layer_1, activation=sigmoid)

neural_network = NeuralNetwork([layer_1])

neural_network.train(inputs=inputs, epochs=100, targets=targets, alpha=0.1)

print("Max loss: ",np.max(neural_network.losses))
print("Min loss: ", np.min(neural_network.losses))



