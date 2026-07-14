# STEP 21: Implement Softmax output and Implement Cross-Entropy Loss
import numpy as np


class Activation_Sigmoid:
    def forward(self,x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self,x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)


class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output


class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        if len(y_true.shape) == 1:
            correct_confidences = y_pred[
                range(samples),
                y_true
            ]

        else:
            correct_confidences = np.sum(
                y_pred * y_true,
                axis=1
            )

        return -np.log(correct_confidences)


class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):

        self.output = self.activation.forward(inputs)

        return self.loss.calculate(
            self.output,
            y_true
        )

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(
                y_true,
                axis=1
            )

        self.dinputs = dvalues.copy()

        self.dinputs[
            range(samples),
            y_true
        ] -= 1

        self.dinputs = self.dinputs / samples

        return self.dinputs


class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias


class Layer:

    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation
        self.grad_weights = None
        self.grad_bias = None

    def get_weight_matrix(self):
        return np.array(
            [neuron.weights for neuron in self.neurons]
        )

    def get_bias_vector(self):
        return np.array(
            [neuron.bias for neuron in self.neurons]
        )

    def set_weight_matrix(self, weights):
        for i, neuron in enumerate(self.neurons):
            neuron.weights = weights[i]

    def set_bias_vector(self, biases):
        for i, neuron in enumerate(self.neurons):
            neuron.bias = biases[i]

    def forward(self, inputs):

        self.inputs = inputs

        weights = self.get_weight_matrix()
        biases = self.get_bias_vector()

        self.z = np.dot(
            inputs,
            weights.T
        ) + biases

        self.output = self.activation.forward(self.z)
        return self.output


    def update_parameters(self, learning_rate):

        weights = self.get_weight_matrix()
        biases = self.get_bias_vector()

        weights -= learning_rate * self.grad_weights.T
        biases -= learning_rate * self.grad_bias

        self.set_weight_matrix(weights)
        self.set_bias_vector(biases)


class NeuralNetwork:

    def __init__(self, layers, loss, learning_rate=0.5, epochs=1000):

        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.loss_history = []
        self.accuracy_history = []


    def forward(self, inputs):

        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output


    def train(self, inputs, targets):

        for epoch in range(self.epochs):

            output = self.forward(inputs)

            self.forward_inputs = output

            loss = self.loss.forward(
                output,
                targets
            )

            self.loss_history.append(loss)

            self.backward(
                targets,
                output
            )

            self.compute_gradients()

            self.update_parameters()

            accuracy = self.calculate_accuracy(
                targets
            )

            self.accuracy_history.append(
                accuracy
            )

    def backward(self, targets, output):

        self.deltas = [None] * len(self.layers)

        self.deltas[-1] = self.loss.backward(
            output,
            targets
        )

        for i in range(len(self.layers)-2, -1, -1):

            current = self.layers[i]
            next_layer = self.layers[i+1]

            self.deltas[i] = (
                self.deltas[i+1]
                @ next_layer.get_weight_matrix()
            ) * current.activation.backward(
                current.z
            )


    def compute_gradients(self):

        for i, layer in enumerate(self.layers):

            layer.grad_weights = (
                layer.inputs.T
                @ self.deltas[i]
            ) / len(layer.inputs)

            layer.grad_bias = np.mean(
                self.deltas[i],
                axis=0
            )


    def update_parameters(self):

        for layer in self.layers:
            layer.update_parameters(
                self.learning_rate
            )


    def calculate_accuracy(self, targets):

        predictions = np.argmax(
            self.forward_inputs,
            axis=1
        )

        labels = np.argmax(
            targets,
            axis=1
        )

        return np.mean(
            predictions == labels
        ) * 100


    def predict(self, inputs):

        output = self.forward(inputs)

        return np.argmax(
            output,
            axis=1
        )


inputs = np.array([
    [1,0],
    [0,1],
    [1,1]
])


targets = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])


rng = np.random.default_rng(42)


hidden_layer = Layer(
    [
        Neuron(rng.uniform(-1,1,2), rng.uniform(-1,1))
        for _ in range(2)
    ],
    Activation_Sigmoid()
)


output_layer = Layer(
    [
        Neuron(rng.uniform(-1,1,2), rng.uniform(-1,1))
        for _ in range(3)
    ],
    Activation_Softmax()
)


loss = Activation_Softmax_Loss_CategoricalCrossentropy()


nn = NeuralNetwork(
    [hidden_layer, output_layer],
    loss
)


nn.train(
    inputs,
    targets
)


print(nn.accuracy_history[-1])

print(
    nn.predict(
        np.array([
            [1,0],
            [0,1],
            [1,1]
        ])
    )
)