import numpy as np
import matplotlib.pyplot as plt


# TODO add support for SoftmaxWithLoss
# TODO add save and load function
class Model:
    def __init__(self, layers=[]):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, *, loss, optimizer, metrics):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.input_layer = Layers.InputLayer()

        for i, layer in enumerate(self.layers):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < len(self.layers) - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss

    def forward(self, X):
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.output)
        return layer.output

    def backward(self, output, y):
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X, y, *, epochs=1, printevery=100, plot=False):
        loss_array = []
        acc_array = []
        lr_array = []
        for epoch in range(1, epochs + 1):
            output = self.forward(X)

            loss = self.loss.calculate(output, y)
            self.metrics.calculate(output, y)
            acc = self.metrics.acc
            loss_array.append(loss)
            acc_array.append(acc)
            lr_array.append(self.optimizer.current_lr)

            self.backward(output, y)

            for layer in self.layers:
                if hasattr(layer, "weights"):
                    self.optimizer.update_params(layer)
            self.optimizer.decay_lr()

            if not epoch % printevery:
                print(f"Epoch: {epoch}\tLoss: {loss}\tAccuracy: {acc}")

        if plot:
            fig, axs = plt.subplots(3)
            axs[0].set_title("Loss")
            axs[0].plot(loss_array)
            axs[1].set_title("Accuracy")
            axs[1].plot(acc_array)
            axs[2].set_title("Learning Rate")
            axs[2].plot(lr_array)
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()
            plt.show()

    def predict(self, input):
        self.input_layer.forward(input)
        for layer in self.layers:
            layer.forward(layer.prev.output)
        return layer.output


# TODO infer number of inputs from previous layer
class Layers:
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

    class Flatten:
        def forward(self, inputs):
            self.inputs = inputs
            self.output = inputs.reshape(inputs.shape[0], -1)

        def backward(self, dvalues):
            self.dinputs = dvalues


class Activations:
    class ReLU:
        def forward(self, inputs):
            self.inputs = inputs
            self.output = np.maximum(0, inputs)

        def backward(self, dvalues):
            self.dinputs = dvalues.copy()
            self.dinputs[self.inputs <= 0] = 0

    class Step:
        def forward(self, inputs):
            self.output = np.heaviside(inputs, 1)

    class Linear:
        def forward(self, inputs):
            self.output = inputs

    class Sigmoid:
        def forward(self, inputs):
            self.output = 1 / (1 + np.exp(-1 * inputs))

    class Softmax:
        def forward(self, inputs):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        def backward(self, dvalues):
            self.dinputs = np.empty_like(dvalues)

            for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
                single_output = single_output.reshape(-1, 1)
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    class SoftmaxWithLoss:
        def __init__(self):
            self.activation = Activations.Softmax()
            self.loss_func = Losses.CategoricalCrossentropy()

        def forward(self, inputs, y_true):
            self.activation.forward(inputs)
            self.output = self.activation.output
            self.loss = self.loss_func.calculate(self.output, y_true)

        def backward(self, dvalues, y_true):
            samples = len(dvalues)

            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)

            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples


class Losses:
    class Loss:
        def calculate(self, output, y):
            sample_losses = self.forward(output, y)
            return np.mean(sample_losses)

    class CategoricalCrossentropy(Loss):
        def forward(self, y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

            # for sparse
            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[range(samples), y_true]
            # for one-hot
            elif len(y_true.shape) == 2:
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            else:
                raise Exception("Check your labels shape")

            return -np.log(correct_confidences)

        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            labels = len(dvalues[0])

            # for sparse
            if len(y_true.shape) == 1:
                y_true = np.eye(labels)[y_true]

            #TODO fix divide by zero error
            self.dinputs = - y_true / dvalues
            self.dinputs = self.dinputs / samples


class Metrics:
    class Accuracy:
        def calculate(self, y_pred, y_true):
            preds = np.argmax(y_pred, axis=1)

            # if one-hot convert them
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)

            self.acc = np.mean(preds == y_true)


class Optimizers:
    class SGD:
        def __init__(self, *, lr=1., decay=0., momentum=0.):
            self.lr = lr
            self.current_lr = lr
            self.decay = decay
            self.step = 0
            self.momentum = momentum

        def update_params(self, layer):
            if self.momentum:

                if not hasattr(layer, "weight_momentums"):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)

                weight_updates = self.momentum * layer.weight_momentums + self.current_lr * layer.dweights
                layer.weight_momentums = weight_updates
                bias_updates = self.momentum * layer.bias_momentums + self.current_lr * layer.dbiases
                layer.bias_momentums = bias_updates
            else:
                weight_updates = self.current_lr * layer.dweights
                bias_updates = self.current_lr * layer.dbiases

            layer.weights -= weight_updates
            layer.biases -= bias_updates

        def decay_lr(self):
            if self.decay:
                self.current_lr = self.lr * (1. / (1. + self.decay * self.step))
                self.step += 1
