

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output
        
