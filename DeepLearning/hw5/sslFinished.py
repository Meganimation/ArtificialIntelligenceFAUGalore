# 11. (coding) Self-supervised learning

import numpy as np

# tiny image
x0 = np.array([
    [0,0,0,0],
    [0,1,0,0],
    [0,0,2,0],
    [0,0,0,0],
], dtype=np.float32)

# 4 rotations (labels 0,1,2,3)
# 0: degree 0, 1: 90 degree, 2: 180 degree, 3: 270 degree

# Compute X and y
# X.shape = (4, 16)
# X = [ x0, x0 rotated 90, x0 rotated 180, x0 rotated 270 ]
# X(i,j) -> X(n-1-j, i)
X = np.stack([np.rot90(x0, k).reshape(-1) for k in range(4)], axis=0)

# y: target (rotation degree)
y = np.array([0, 1, 2, 3])

# tiny linear model
rng = np.random.default_rng(0)
# initialize W and b
W = (0.01 * rng.normal(size=(16, 4))).astype(np.float32)
b = np.zeros(4, dtype=np.float32)

# forward
logits = X @ W + b

# X : (4,16), W : (16,4), b : (4,)
# logits.shape = (4,4)
# Convert logits into probabilities P using softmax
# P.shape = (4,4), P_ij = P(rotation=j | Xi)
logits -= logits.max(axis=1, keepdims=True)
P = np.exp(logits)
P /= P.sum(axis=1, keepdims=True)

# loss (cross-entropy): loss = -log P(y|x)
loss = -np.log(P[np.arange(4), y] + 1e-12).mean()
print("loss before:", float(loss))

# -------- single SGD step --------
# change Y to one-hot vector
Y = np.zeros_like(P)
Y[np.arange(4), y] = 1.0

# compute gradient
dlogits = (P - Y) / 4.0   # gradient of cross-entropy after softmax
dW = X.T @ dlogits        # [16,4]
db = dlogits.sum(axis=0)

lr = 0.5
W -= lr * dW
b -= lr * db
# --------------------------------

# forward again
logits = X @ W + b
logits -= logits.max(axis=1, keepdims=True)
P = np.exp(logits)
P /= P.sum(axis=1, keepdims=True)

loss_after = -np.log(P[np.arange(4), y] + 1e-12).mean()
print("loss after :", float(loss_after))
