Hardcode a single neuron (3 inputs → 1 output)
Add a bias term
Move inputs and weights into vectors (lists/arrays)
Compute the dot product using a loop
Create a reusable Neuron class
Test the neuron with multiple input samples
Implement activation functions (Step, Sigmoid, ReLU)
Create a Layer class containing multiple neurons
Perform forward propagation through one layer
Replace loops with matrix multiplication (NumPy)
Process a batch of inputs at once
Create a NeuralNetwork class with multiple layers
Implement forward propagation through the entire network
Create a small dataset (e.g., AND, OR)
Implement a loss function (Mean Squared Error)
Calculate prediction error
Adjust weights manually to observe the effect
Implement numerical gradient estimation (finite differences)
Learn partial derivatives and gradients
Derive gradients for a single neuron
Implement gradient descent
Update weights automatically
Train a single neuron until the loss decreases
Implement backpropagation for one layer
Extend backpropagation to multiple layers
Train on the XOR dataset (observe failure with one layer)
Add a hidden layer
Train until XOR is solved
Refactor into reusable modules (Neuron, Layer, Network, Loss, Optimizer)
Add configurable hyperparameters (learning rate, epochs, batch size)
Add training metrics and loss visualization
Save and load model weights
Implement mini-batch gradient descent
Add different activation functions (Tanh, Leaky ReLU)
Implement Softmax output
Implement Cross-Entropy Loss
Train on a real dataset (e.g., Iris)
Train on handwritten digits (MNIST)
Add regularization (L2, Dropout)
Implement different optimizers (Momentum, RMSProp, Adam)
Build a small neural network framework from scratch
Compare your implementation with a framework like PyTorch to understand how each abstraction maps to what you built.
