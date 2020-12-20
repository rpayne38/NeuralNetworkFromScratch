import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class InputLayer:
    def forward(self, input):
        self.output = input

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.input_layer = InputLayer()

        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < len(self.layers) - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss

    def forward(self, X):
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.output)

    def backward(self, output, y):
        self.loss.backward(output, y)
        for layer in self.layers:
            if hasattr(layer, "weights"):
                layer.backward(layer.next.dinputs)

    def train(self, X, y, *, epochs, printfreq):
        for epoch in range(1, epochs+1):
            self.forward(X)

            self.backward(s)

