import numpy as np

data = [(1, 2), (2, 3), (3, 5), (4, 3)]
x = np.array([p[0] for p in data], dtype=float)
y = np.array([p[1] for p in data], dtype=float)
X = np.column_stack([np.ones_like(x), x])

XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
beta = XtX_inv @ X.T @ y
w0, w1 = beta[0], beta[1]

x_new = 5.0
y_pred = w0 + w1 * x_new

y_hat = X @ beta
e = y - y_hat


print("errors:", e)
