import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

true_mean = 5.0
true_std = 2.0
no_data = 1000
orig_data = np.random.normal(true_mean, true_std, no_data)

mle_mean = np.mean(orig_data)

mle_std = np.std(orig_data, ddof=0) 

print(f"True Mean is {true_mean}, True Std is {true_std}")
print(f"MLE Mean is {mle_mean}, MLE Std is {mle_std}")

x = np.linspace(min(orig_data) - 1, max(orig_data) + 1, no_data)
true_pdf = norm.pdf(x, true_mean, true_std)
mle_pdf = norm.pdf(x, mle_mean, mle_std)

plt.figure(figsize=(10, 6))
plt.hist(orig_data, bins=30, density=True, alpha=0.5, label='Data histogram')
plt.plot(x, true_pdf, linestyle='--', linewidth=2, color='blue', label=f'True PDF (μ={true_mean}, σ={true_std})')
plt.plot(x, mle_pdf, linewidth=2, color='red', label=f'MLE PDF (μ={mle_mean:.2f}, σ={mle_std:.2f})')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Gaussian Distribution: True vs MLE Estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()