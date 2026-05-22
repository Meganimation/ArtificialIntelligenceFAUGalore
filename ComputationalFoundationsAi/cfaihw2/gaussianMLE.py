import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate some data from a Gaussian distribution
# ‘’’
# YOUR WORK HERE
# Generate data point based on Gaussian distribution.
# Use np.random.normal with the following parameters 
# true_mean: mean of Gaussian
# true_std: standard dev of Gaussian
# no_data: no of data points (>100)
# orig_data = np.random.normal(OOO)
# ‘’’

#  ‘’’
# YOUR WORK HERE
# # MLE estimation
# Compute mle estimate of mean from orig_data
# Compute mle estimate of std from orig_data
# # display the values
# Show the values of true_mean, true_std, mle_mean, mle_std
# ‘’’

# Visualize the results
x = np.linspace(min(orig_data) - 1, max(orig_data) + 1, no_data)
true_pdf = norm.pdf(x, true_mean, true_std)
mle_pdf = norm.pdf(x, mle_mean, mle_std)

# ‘’’
# 12. 1)-3 (6 pts)
# YOUR WORK HERE
# Draw graphs of true_pdf and mle_pdf together
# For example, 
# plt.plot(x,true_pdf, linestyle=’--‘…)
# plt.plot(x,mle_pdf, …)
# ‘’’
