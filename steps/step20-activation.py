# STEP 20: Add different activation functions (Tanh, Leaky ReLU)
import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x)/((1+np.exp(-x))**2)
    else:
        return 1 / (1 + np.exp(-x))

def relu(x,derivate=False):
    if derivate:
        return np.where(x>0,1,0)
    else:
        return  np.maximum(0,x)
    
def leaky_relu(x, alpha=0.01, derivate=False):
    if derivate:
        return np.where(x>0, 1, alpha)
    else:
        return np.where(x>0,x, alpha*x)
    
def tanh(x,derivate=False):
    tanhx = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    if derivate:
        return 1 - tanhx**2
    else:
        return tanhx

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    def forward(self,y_true, y_pred):
        return np.mean(( y_pred- y_true ) ** 2)

    def backward(self,y_true,y_pred):
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

    def get_weight_matrix(self):
        return np.array([neuron.weights for neuron in self.neurons])
    
    def get_bias_vector(self):
        return np.array([neuron.bias for neuron in self.neurons])
    
    def set_weight_matrix(self,new_layer_weights):
        for index,neuron in enumerate(self.neurons):
            neuron.weights = new_layer_weights[index]

        
    def set_bias_vector(self, new_layer_bias):
        for index,neuron in enumerate(self.neurons):
            neuron.bias = new_layer_bias[index]
          

    def forward(self,inputs ):
        self.inputs = inputs
        weight_matrix = self.get_weight_matrix()
        bias_vector = self.get_bias_vector()
        z = np.dot(inputs , weight_matrix.T) +  bias_vector 
        self.z = z
        return self.activation(z)
    
    def update_parameters(self,learning_rate):
        weight_matrix = self.get_weight_matrix()
        bias_vector = self.get_bias_vector()
        
        
        new_layer_weights = weight_matrix - learning_rate * self.grad_weights.T
        new_layer_bias = bias_vector - learning_rate * self.grad_bias.T
        
        self.set_weight_matrix(new_layer_weights)
        self.set_bias_vector(new_layer_bias)
    
    


    
class NeuralNetwork:

    def __init__(self, layers,loss_function, learning_rate=0.5, epochs=10000):
        self.layers = layers
        self.loss_function= loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.outputs =[]
        self.loss_history = []
        self.accuracy_history =[]
        self.deltas = [None]*len(layers)
        
    
    def forward(self, inputs):
        layer_input = inputs
        layer_output = 0
        for layer in self.layers:
            layer_output = layer.forward(layer_input)
            layer_input = layer_output
        return layer_output

    def train(self, inputs, targets):
        for epoch in range(self.epochs):
            result = self.forward(inputs)
            self.outputs.append(result)
            loss_history = self.loss_function.forward(targets, result)
            self.loss_history.append(loss_history)

            self.backward(targets)
            self.compute_gradients()
            self.update_parameters()
            accuracy = self.calculate_accuracy(targets)
            self.accuracy_history.append(accuracy)

    def backward(self,targets):
        result = self.outputs[-1]
        loss_derivate = self.loss_function.backward(targets, result)
        last_layer = self.layers[-1]
        self.deltas[-1] = loss_derivate * last_layer.activation(last_layer.z,derivate=True)
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            hidden_layer_delta = (self.deltas[i+1] @ next_layer.get_weight_matrix()) * current_layer.activation(current_layer.z, derivate=True)
            self.deltas[i] = hidden_layer_delta
            

    def compute_gradients(self):
        for index,layer in enumerate(self.layers):
                layer.grad_weights = ( layer.inputs.T @ self.deltas[index])/len(layer.inputs)
                layer.grad_bias = np.mean(self.deltas[index],axis=0)

    def update_parameters(self):   
        for layer in self.layers:
            layer.update_parameters(self.learning_rate)
    
    def calculate_accuracy(self, targets):
        result = self.outputs[-1]
        rounded = np.round(result)
        correct = np.count_nonzero(rounded == targets)
        return correct / len(targets) * 100

    def predict(self, input):
        return self.forward(input)
    



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



activation_functions = [sigmoid, relu, leaky_relu, tanh]

for activation_function in activation_functions:

    rng = np.random.default_rng(42)

    hidden_neurons = [
        Neuron(rng.uniform(-1, 1, 2), rng.uniform(-1, 1))
        for _ in range(2)
    ]

    output_neurons = [
        Neuron(rng.uniform(-1, 1, 2), rng.uniform(-1, 1))
    ]

    hidden_layer = Layer(hidden_neurons, activation=activation_function)
    output_layer = Layer(output_neurons, activation=sigmoid)

    mse_loss = MSE()
    nn = NeuralNetwork([hidden_layer, output_layer], mse_loss)



    nn.train(inputs, targets)


    print("Function Name:", activation_function.__name__)
    print("Accuracy:", nn.accuracy_history[-1], "%")
    test_cases = [
        ([1, 1], 0),
        ([0, 0], 0),
        ([1, 0], 1),
        ([0, 1], 1)
    ]
    for test_input, expected in test_cases:
        raw_pred = nn.predict(test_input)
        rounded_pred = int(np.round(raw_pred)[0])
        print(f"Input: {test_input} -> Predicted: {rounded_pred} (Expected: {expected})")

