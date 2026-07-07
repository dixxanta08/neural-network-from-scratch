STEP 01 : Hardcode a single neuron (3 inputs → 1 output)
STEP 02 : Add a bias term
STEP 03a : Move inputs and weights into vectors and dot product using list comprehension (lists/arrays)
STEP 03b : Use Numpy and numpy dot function
STEP 04a : Create a reusable Neuron class
STEP 04b : Test the neuron with multiple input samples
Step 05 : Implement activation functions (Step, Sigmoid, ReLU)
STEP 06 : Create a Layer class containing multiple neurons and perform forward propagation through one layer
STEP 07 : Replace loops with matrix multiplication (NumPy)
STEP 08 : Process a batch of inputs at once
STEP 09 : Create a NeuralNetwork class with multiple layers and Implement forward propagation through the entire network
STEP 10 : Create a small dataset (e.g., AND, OR)
STEP 11 : Implement a loss function (Mean Squared Error) and calculate prediction error also adjust weights manually to observe the effect
STEP 12 : Implement numerical gradient estimation (finite differences). Next, Learn partial derivatives and gradients
STEP 13 : Derive gradients for a single neuron and Implement gradient descent
STEP 14 : Update weights automatically and Train a single neuron until the loss decreases
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
