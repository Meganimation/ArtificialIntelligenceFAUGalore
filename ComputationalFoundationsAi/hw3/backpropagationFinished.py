'''
Network structure
- 2 input neurons
- 2 hidden neurons
- 1 output neuron
- Uses sigmoid activation
- Trains on one example using mean squared error
'''

import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    # ‘’’
    # returns sigmoid value of x
    # ‘’’
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # ‘’’
    # returns the derivative of sigmoid x
    # ‘’’
    return x * (1 - x)

# For simplicity, process 1 training example
# Shape of X=(1, 2)
X = np.array([[0.5, 0.1]])  

# True output
# Shape of y=(1, 1)
y = np.array([[1]])  

# Initialize weights and biases with random values
input_size = 2
hidden_size = 2
output_size = 1

# Weights
# W1: weights between input layer & hidden layer
# shape of W1=(2, 2)
W1 = np.random.rand(input_size, hidden_size)   

# W2: weights between hidden layer & output layer
# shape of W2=(2, 1)
W2 = np.random.rand(hidden_size, output_size)

# Biases
# b1: biases of hidden nodes
# shape of b1=(1, 2)
b1 = np.random.rand(1, hidden_size)  

# b2: biase of output nodes
# shape of b1=(1, 1)
b2 = np.random.rand(1, output_size)

# learning rate
learning_rate = 0.1
# number of iterations
epochs = 2000

# backpropagation
for epoch in range(epochs):
    # Forward computing
    # compute the output of hidden nodes and output node
    # shape of hidden_input=(1, 2)
    hidden_input = np.dot(X, W1) + b1      
    # shape of hidden_output=(1, 2)
    hidden_output = sigmoid(hidden_input)  # (1, 2)

    # ‘’’
    # compute ‘final_out’ (output of output node)
    # shape of final_input=(1, 1)
    # using final_output, compute class prediction ‘y_pred’ (1 or 0)
    # compute error using squared error
    # ‘’’
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    y_pred = (final_output >= 0.5).astype(int)

    error = y - final_output
    loss = np.mean(error ** 2)

    # ‘’’
    # compute dW2 and db2 using the formula in p. 25-26 in NN slides.
    # # update W2 and b2 (output weights)
    # W2 -= learning_rate * dW2
    # b2 -= learning_rate * db2
    # compute dW1 and b1 using the formula in p. 27-30 in NN slides.
    # # update W1 and b1 (hidden weights)
    # W1 -= learning_rate * dW1
    # b1 -= learning_rate * db1
    # ‘’’
    # Output layer gradients
    delta_output = -(y - final_output) * sigmoid_derivative(final_output)
    dW2 = np.dot(hidden_output.T, delta_output)
    db2 = np.sum(delta_output, axis=0, keepdims=True)

    # Hidden layer gradients
    delta_hidden = np.dot(delta_output, W2.T) * sigmoid_derivative(hidden_output)
    dW1 = np.dot(X.T, delta_hidden)
    db1 = np.sum(delta_hidden, axis=0, keepdims=True)

    # Gradient descent updates
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print loss value
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final prediction value
print("\nFinal prediction:", y_pred)
