import numpy as np

# objective function: f(w) = (w - 3)^2 

def f(w):
    return (w - 3) ** 2

def grad_f(w):
    # derivative of (w - 3)^2 is 2(w - 3)
    return 2 * (w - 3)


w = 10.0                   
lr = 0.1
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

m = 0.0                    # 1st moment (mean)
v = 0.0                    # 2nd moment (uncentered variance)
t = 0

for step in range(1, 51):
    t += 1
    g = grad_f(w)

    # moment updates
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)

    # bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # parameter update
    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    if step in [1, 2, 3, 5, 10, 20, 50]:
        print(f"step {step:2d} | w={w:.6f} | f(w)={f(w):.6f} | grad={g:.6f}")
