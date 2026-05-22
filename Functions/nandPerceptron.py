import numpy as np
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([1, 1, 1, 0])

weights = np.zeros(2)  # [w1, w2]
bias = 0.0
learning_rate = 0.1
epochs = 20
print("Starting Training...")
for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias

        prediction = 1 if linear_output > 0 else 0

        update = learning_rate * (y[i] - prediction)
        if update != 0:
            weights += update * X[i]
            bias += update
            errors += 1
            
    if errors == 0:
        print(f"Converged at epoch {epoch}!")
        break

print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")