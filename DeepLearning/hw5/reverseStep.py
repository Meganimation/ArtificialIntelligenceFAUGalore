
# 12. (coding) reverse step in diffusion model

import numpy as np

np.random.seed(0)

# -----------------------------------
# diffusion schedule
# -----------------------------------
T = 10
beta = np.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)

# choose a timestep
t = 5

beta_t = beta[t]
alpha_t = alpha[t]
alpha_bar_t = alpha_bar[t]

# -----------------------------------
# sample x_t (noisy data)
# -----------------------------------
# small 2-D example
x_t = np.array([1.5, -0.3])   

# predicted noise from neural network
eps_theta = np.array([0.2, -0.1])

# noise for stochastic sampling
# 12.-1: sample z from standard normal (same shape as x_t)
z = np.random.randn(*x_t.shape)

# -----------------------------------
# reverse diffusion step
# -----------------------------------
sigma_t = np.sqrt(beta_t)
# 12.-2: compute mean mu_theta(x_t, t) then sample x_prev
# mu = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta)
mu = (1 / np.sqrt(alpha_t)) * (x_t - (beta_t / np.sqrt(1 - alpha_bar_t)) * eps_theta)
x_prev = mu + sigma_t * z

print("x_t      =", x_t)
print("pred eps =", eps_theta)
print("noise z  =", z)
print("x_{t-1}  =", x_prev)


# 13. (extra pts) (coding) forward step in diffusion model

import numpy as np

np.random.seed(0)

# -----------------------------
# noise schedule
# -----------------------------
T = 10
beta = np.linspace(0.0001, 0.02, T)
alpha = 1 - beta

# choose timestep
t = 3
beta_t = beta[t]
alpha_t = alpha[t]

# -----------------------------
# original data x_(t-1)
# -----------------------------
x_prev = np.array([1.0, -0.5])   # small 2D example

# random Gaussian noise
# 13.-1: sample eps from standard normal, compute x_t via reparameterization
# x_t = sqrt(alpha_t) * x_prev + sqrt(beta_t) * eps
eps = np.random.randn(*x_prev.shape)
x_t = np.sqrt(alpha_t) * x_prev + np.sqrt(beta_t) * eps

# -----------------------------
# forward diffusion step
# -----------------------------

print("x_(t-1) =", x_prev)
print("noise ε =", eps)
print("x_t     =", x_t)
