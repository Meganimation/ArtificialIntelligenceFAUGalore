
12. (coding) reverse step in diffusion model

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
‘’’
12.-1 [4 pts] 
YOUR WORK HERE
Compute noise z from standard normal distribution
‘’’

# -----------------------------------
# reverse diffusion step
# -----------------------------------
sigma_t = np.sqrt(beta_t)
‘’’
12.-2 [8 pts]
YOUR WORK HERE
Compute x_prev from x_t using
 
X_(t-1)~N(μ_θ (X_t,t),σ_t^2 I)⇒μ_θ (X_t,t)+σ_t z
‘’’

print("x_t      =", x_t)
print("pred eps =", eps_theta)
print("noise z  =", z)
print("x_{t-1}  =", x_prev)


13. (extra pts) (coding) forward step in diffusion model

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
‘’’
13.-1 [6 pts]
YOUR WORK HERE
Compute noise eps from standard normal distribution
Compute x_t from x_prev: N(√(α_t ) X_(t-1),β_t I)
Use reparameterization technique
‘’’

# -----------------------------
# forward diffusion step
# -----------------------------

print("x_(t-1) =", x_prev)
print("noise ε =", eps)
print("x_t     =", x_t)
