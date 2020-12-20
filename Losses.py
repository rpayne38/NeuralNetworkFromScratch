import numpy as np


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

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # for sparse
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = - y_true / dvalues
        self.dinputs = self.dinputs / samples

# model
dense1 = Layers.Dense(2, 64)
relu = Activations.ReLU()
dense2 = Layers.Dense(64, 3)
softmax = Activations.SoftmaxWithLoss()

lr = 1.
accuracy = Metrics.Accuracy()
optimizer = Optimizers.SGD(lr=lr, decay=1e-3, momentum=0.9)

loss_array = []
acc_array = []

for epoch in range(3000):
    # forward pass
    dense1.forward(X)
    relu.forward(dense1.output)
    dense2.forward(relu.output)
    softmax.forward(dense2.output, y)
    accuracy.calculate(softmax.output, y)

    # backward pass
    softmax.backward(softmax.output, y)
    dense2.backward(softmax.dinputs)
    relu.backward(dense2.dinputs)
    dense1.backward(relu.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.decay_lr()

    loss_array.append(softmax.loss)
    acc_array.append(accuracy.acc)

    if not epoch % 100:
        print(f"Epoch: {epoch}\tLoss: {softmax.loss}\tAccuracy: {accuracy.acc}")

print(optimizer.current_lr)

fig, axs = plt.subplots(2)
axs[0].set_title("Loss")
axs[0].plot(loss_array)
axs[1].set_title("Accuracy")
axs[1].plot(acc_array)
axs[0].grid()
axs[1].grid()
plt.show()