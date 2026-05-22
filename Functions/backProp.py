import numpy as np

eta = 0.2  
weights_initial = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(o):
    return o * (1 - o)
x = np.array([[1, 0]]) 
target = 0
W1 = np.full((2, 2), weights_initial)  # Input to Hidden
B1 = np.full((1, 2), weights_initial)
W2 = np.full((2, 1), weights_initial)  # Hidden to Output
B2 = np.full((1, 1), weights_initial)

net_Z = np.dot(x, W1) + B1
out_Z = sigmoid(net_Z)

net_Y = np.dot(out_Z, W2) + B2
out_Y = sigmoid(net_Y)

print(f"--- Forward Pass Results for x=[1, 0] ---")
print(f"Net Input Z: {net_Z[0]}")
print(f"Output Z1:   {out_Z[0][0]:.4f}")
print(f"Output Z2:   {out_Z[0][1]:.4f}")
print(f"Net Input Y: {net_Y[0][0]:.4f}")
print(f"Output Y1:   {out_Y[0][0]:.4f}")

error = 0.5 * (target - out_Y[0][0])**2
print(f"\nInitial SSE: {error:.4f}")