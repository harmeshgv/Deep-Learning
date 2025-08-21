import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix

class SLP:
    def __init__(self, input_size,learning_rate = 0.01, epochs = 10 ):
        self.weights = np.random.uniform(-0.01, 0.01, size = input_size)
        self.bais = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def _activation(self, x):
        return np.where(x >= 0 , 1, -1)

    def _predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bais
        return self._activation(linear_output)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                output = self._predict(xi)
                error = target - output

                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bais += self.learning_rate * error


iris = load_iris()
X = iris.data
Y = iris.target
y = np.where(Y == 0, -1, 1) 

print(X)
print(y)

SimpleLinearPerceptron = SLP(input_size=2 , learning_rate=0.0001, epochs=200)
SimpleLinearPerceptron.fit(X, y)
print(SimpleLinearPerceptron.bais)
print(SimpleLinearPerceptron.weights)


preds = []
for sample in X:
    preds.append(SimpleLinearPerceptron._predict(sample))

print(classification_report(y, preds))



    