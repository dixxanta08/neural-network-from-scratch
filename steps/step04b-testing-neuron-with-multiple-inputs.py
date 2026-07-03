# Step 04(b): Test the neuron with multiple input samples
import numpy as np
from nn.neuron import Neuron

inputs_1 = np.array([0.9, 0.3, 0.2])
weights_1 = np.array([0.6, 0.8, 0.3])
bias_1 = 0.5

inputs_2 = np.array([-0.5, 0.0, 0.8])
weights_2 = np.array([0.5, -1.2, 0.4])
bias_2 = -0.2

inputs_3 = np.array([1.0, 0.0, 1.0])
weights_3 = np.array([-0.1, 0.5, 0.9])
bias_3 = 0.0


neuron_1 = Neuron(weights_1, bias_1)
print("Test Case 1 Output:", neuron_1.forward(inputs_1))

neuron_2 = Neuron(weights_2, bias_2)
print("\nTest Case 2 Output:", neuron_2.forward(inputs_2))

neuron_3 = Neuron(weights_3, bias_3)
print("\nTest Case 3 Output:", neuron_3.forward(inputs_3))

