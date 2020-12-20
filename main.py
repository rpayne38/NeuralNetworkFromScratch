import numpy as np
from PyNet import Layers, Optimizers, Activations, Metrics, Losses, Model
import struct

np.random.seed(0)

with open('/home/rob/Downloads/train-images.idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    X = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    X = X.reshape((size, nrows, ncols))

with open('/home/rob/Downloads/train-labels.idx1-ubyte', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    y = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))

model = Model([
    Layers.Flatten(),
    Layers.Dense(28*28, 32),
    Activations.ReLU(),
    Layers.Dense(32, 64),
    Activations.ReLU(),
    Layers.Dense(64, 10),
    Activations.Softmax()
])

model.compile(loss=Losses.CategoricalCrossentropy(),
              optimizer=Optimizers.SGD(lr=0.01, decay=0.1, momentum=0.9),
              metrics=Metrics.Accuracy())

model.train(X, y, epochs=100, printevery=10, plot=True)

