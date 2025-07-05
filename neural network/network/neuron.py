import data_generator as Gen
import numpy as np

# 2 -> 16 -> 16 -> 3
# input -> hidden -> hidden -> output

# Data Gathering
X, y = Gen.create_data(100, 3)

X = np.array(X)
y = np.array(y)

X_train = np.concatenate((X[1:75], X[101:175], X[201:275]))
y_train = np.concatenate((y[1:75], y[101:175], y[201:275]))


X_test = np.concatenate((X[76:100], X[176:200], X[276:300]))
y_test = np.concatenate((y[76:100], y[176:200], y[276:300]))

W1 = 0.1 * np.random.randn(2, 16)
W2 = 0.1 * np.random.randn(16, 16)
W3 = 0.1 * np.random.randn(16, 3)
B1 = np.zeros((1, 16))
B2 = np.zeros((1, 16))
B3 = np.zeros((1, 3))


# Forward Propagation
def ReLU(inputs):
    return np.max(9, inputs)


def Softmax(inputs):
    return np.exp(inputs) / np.sum(np.exp(inputs))


class Layer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        pass

    def forward_prop(self, inputs=X_train, Relu=True):
        self.output = (
            np.dot(inputs, self.weights) + self.biases
        )  # 74 * 2 -> 2 * 16 = 74 * 16 (training output)
        if Relu:
            self.output = ReLU(self.output)
        else:
            self.output = Softmax(self.output)


hiddenLayer1 = Layer(W1, B1)
hiddenLayer2 = Layer(W2, B2)
outputLayer = Layer(W3, B3)


hiddenLayer1.forward_prop()

hiddenLayer1.output
