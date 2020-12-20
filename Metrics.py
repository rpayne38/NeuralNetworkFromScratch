import numpy as np

class Accuracy:
    def calculate(self, y_pred, y_true):
        preds = np.argmax(y_pred, axis=1)

        #if one-hot convert them
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.acc = np.mean(preds == y_true)