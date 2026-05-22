import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

x = np.array([0.5, -1.2, 3.1])  
w = np.array([0.1, 0.2, -0.1])   
target = 1.0                     

z = np.dot(w, x)                 
output = relu(z)                 
error = 0.5 * (target - output)**2
error_signal = -(target - output)
activation_grad = relu_derivative(z)
gradient = error_signal * activation_grad * x

print(f"Net Input (z): {z:.4f}")
print(f"Output (o):    {output:.4f}")
print(f"Gradient (dw): {gradient}")