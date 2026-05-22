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
        # ‘’’
        # Compute ∇_w L(w)={█(0,          if y_i (wx_i+b)≥1@&-y_i x_i, if y_i (wx_i+b)<1)┤
        # Update w using w=w-η∇_w J(w)
        # ‘’’

# ‘’’
# Compute prediction for data X
#     preds=[predicted class values]
# print w, b 
# print accuracy of svm
#  ‘’’

# Plot the data 
# ‘’’
# Plot a graph of X, y dataset something like that
# ‘’’

# ‘’’
# Draw the line of svm using the values of w, b
# ‘’’
