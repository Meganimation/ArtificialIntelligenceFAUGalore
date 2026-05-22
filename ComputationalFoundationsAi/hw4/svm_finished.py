import numpy as np
import matplotlib.pyplot as plt


# Gradient descent SVM with Hinge loss

# learning rate
lr=0.01
# regularization parameter
regu_lambda=0.01
# no of iteration
n_iters=500

# Generate 2D artificial dataset
# linearly separable dataset
np.random.seed(1)
X1 = np.random.randn(50, 2) + np.array([-2, -2])
X2 = np.random.randn(50, 2) + np.array([2, 2])
X = np.vstack((X1, X2))
y = np.hstack((np.ones(50), -1 * np.ones(50)))

n_samples, n_features = X.shape
y_ = y.copy()
    
w = np.zeros(n_features)
b = 0

for _ in range(n_iters):
    for idx, x_i in enumerate(X):
        condition = y_[idx] * (np.dot(x_i, w) + b) >= 1

        if condition:
            dw = 2 * regu_lambda * w
            db = 0
        else:
            dw = 2 * regu_lambda * w - np.dot(x_i, y_[idx])
            db = -y_[idx]

        w = w - lr * dw
        b = b - lr * db

preds = np.sign(np.dot(X, w) + b)
preds[preds == 0] = 1

print("w:", w)
print("b:", b)
print("Accuracy of SVM:", np.mean(preds == y) * 100, "%")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x - b + offset) / w[1]

x0_1 = np.min(X[:, 0]) - 1
x0_2 = np.max(X[:, 0]) + 1

x1_1 = get_hyperplane_value(x0_1, w, b, 0)
x1_2 = get_hyperplane_value(x0_2, w, b, 0)

x1_1_m = get_hyperplane_value(x0_1, w, b, -1)
x1_2_m = get_hyperplane_value(x0_2, w, b, -1)

x1_1_p = get_hyperplane_value(x0_1, w, b, 1)
x1_2_p = get_hyperplane_value(x0_2, w, b, 1)

plt.plot([x0_1, x0_2], [x1_1, x1_2], "k-")
plt.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k--")
plt.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k--")

plt.xlim(x0_1, x0_2)
plt.ylim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)
plt.title("Linear SVM (Gradient Descent with Hinge Loss)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
