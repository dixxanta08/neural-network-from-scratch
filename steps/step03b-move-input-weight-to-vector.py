# Move inputs and weights into vectors (numpy)
# NumPy is used because it makes vector math fast, clean, and scalable for neural networks

import numpy as np

inputs = np.array([0.9, 0.3, 0.2])
weights = np.array([0.6, 0.8, 0.3])

bias = 0.5
 
dot = np.dot(inputs,weights)
output = dot+bias
print(output)