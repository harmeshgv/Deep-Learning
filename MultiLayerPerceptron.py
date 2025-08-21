import torch.nn as nn
import torch
from sklearn.datasets import load_iris, load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim

class MultiLayerPerceeptron(nn.Module):
    def __init__(self,  input_size,hidden_layers,num_classes,  epochs = 100, learning_rate = 0.01):
        super(MultiLayerPerceeptron, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_layers)
        self.output = nn.Linear(hidden_layers, num_classes)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion  = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)

    def forward(self, x):
        x = torch.softmax(self.hidden(x))
        x = self.output(x)
        return x
    
    def fit(self, x, y):
        for i in range(self.epochs):
            for xi, target in zip(x, y):
                xi = xi.unsqueeze(0)
                output = self.forward(xi)
                loss = self.criterion(output, target.unsqueeze(0))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (i+1) % 10 == 0:
                print(f"Epoch [{i+1}/100], Loss: {loss.item():.4f}")


    


iris = load_iris()
iris = load_breast_cancer()
X = iris.data
Y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=43)
print("loaded data")
x_train = torch.tensor(x_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.long)
x_test = torch.tensor(x_test, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype= torch.long)
print("turned tensor")
model = MultiLayerPerceeptron(input_size=30, hidden_layers=12, num_classes=2, epochs=100, learning_rate=0.001)
model.fit(x_train, y_train)

with torch.no_grad():
    outputs = model(x_test)
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {acc.item()*100:.2f}%")

    