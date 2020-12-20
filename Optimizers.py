import numpy as np

class SGD():
    def __init__(self, lr=1., decay=0., momentum=0.):
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
