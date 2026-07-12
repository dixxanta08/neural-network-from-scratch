# STEP 18: Save and Load Model
import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x)/((1+np.exp(-x))**2)
    else:
        return 1 / (1 + np.exp(-x))

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
            print(f"\n==============================\nEPOCH {epoch} of {self.epochs}\n==============================")
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
    
    def save_model(self):
        model_parameters = {}
        for index, layer in enumerate(self.layers):
            model_parameters[f"layer_{index}_weights"] = layer.get_weight_matrix()
            model_parameters[f"layer_{index}_bias"] = layer.get_bias_vector()
        np.savez("model.npz", **model_parameters)

    def load_model(self):
        data = np.load("model.npz")
        for index, layer in enumerate(self.layers):
            weights = data[f"layer_{index}_weights"]
            bias = data[f"layer_{index}_bias"]
            layer.set_weight_matrix(weights)
            layer.set_bias_vector(bias)

    

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

mse_loss = MSE()
nn = NeuralNetwork([hidden_layer, output_layer], mse_loss)

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

nn.train(inputs, targets)




nn.save_model()

print("Before:", nn.predict([1, 1]))

# Change weights manually
for layer in nn.layers:
    layer.set_weight_matrix(np.zeros_like(layer.get_weight_matrix()))
    layer.set_bias_vector(np.zeros_like(layer.get_bias_vector()))

print("After modification:", nn.predict([1, 1]))

nn.load_model()

print("After loading:", nn.predict([1, 1]))
print("Cleaned ouput for [1,1]:", np.round(nn.predict([1, 1])), "expected 0")