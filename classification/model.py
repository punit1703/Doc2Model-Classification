import numpy as np

class Classification:
    def __init__(self,epochs=10000,alpha=1e-5):
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.b = None
        self.mean = None
        self.std = None

    def scale_features(self, x):
        if self.mean is None:
            self.mean = np.mean(x, axis=0)
        if self.std is None:
            self.std = np.std(x, axis=0)

        return (x - self.mean) / self.std

    def one_hot(self, y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def computePartialDerivative(self, x, y_onehot, n):
        z = np.dot(x, self.w) + self.b
        y_pred = self.softmax(z)

        dj_dw = (1 / n) * np.dot(x.T, y_pred - y_onehot)
        dj_db = (1 / n) * np.sum(y_pred - y_onehot, axis=0)

        return dj_dw, dj_db

    def fit(self, x, y):
        x = self.scale_features(x)
        n, f = x.shape
        num_classes = len(np.unique(y))
        y_onehot = self.one_hot(y, num_classes)

        self.w = np.zeros((f, num_classes))
        self.b = np.zeros(num_classes)

        for _ in range(self.epochs):
            dw, db = self.computePartialDerivative(x, y_onehot, n)
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

    def predict(self, x):
        x = self.scale_features(x)
        z = np.dot(x, self.w) + self.b
        probs = self.softmax(z)
        return np.argmax(probs, axis=1)

    def accuracy(self, y_test, y_pred):
        return np.mean(y_test == y_pred)