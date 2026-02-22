import numpy as np

def sigmoid(z):

    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, w):

    m = len(y)
    z = X @ w
    p = sigmoid(z)
    loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
    return loss

def train_logistic(X, y, lr=0.01, epochs=1000):

    m, n = X.shape
    w = np.zeros(n)
    
    for _ in range(epochs):
        z = X @ w
        p = sigmoid(z)
        gradient = (X.T @ (p - y)) / m
        w -= lr * gradient
    
    return w

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

w = train_logistic(X_train_bias, y_train, lr=0.01, epochs=2000)

train_pred = sigmoid(X_train_bias @ w) >= 0.5
test_pred = sigmoid(X_test_bias @ w) >= 0.5

print("Scratch Train Accuracy:", accuracy_score(y_train, train_pred))
print("Scratch Test Accuracy:", accuracy_score(y_test, test_pred))


# obtained results:

# Scratch Train Accuracy: 0.9208791208791208
# Scratch Test Accuracy: 0.9122807017543859