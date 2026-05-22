import numpy as np
import matplotlib.pyplot as plt

# Define the true function
# Suppose the true error function is cosine function
def true_function(X):
    return np.sin(X)+2*cos(X)

# Define the RBF (Radial Basis Function) kernel (Gaussian Kernel)
# Computes the RBF (Gaussian) kernel between two vectors X1 and X2.
# X1 : numpy array 1D (size=n)
# X2 : numpy array 1D (size=m)
# returns:
# K : numpy array of shape (size of X1=n, size of X2=m).

def rbf_kernel(X1, X2):
    ‘’’
    8. 1)-1 (6 pts)
    YOUR WORK HERE
    1) Implement the following RBF kernel K
K(x,x)=[■(k(x_1,x_1)&⋯&k(x_1,x_m)@⋮&⋱&⋮@k(x_n, x_1)&⋯&k(x_n, x_m))]
where k(x_i, x_j )=exp⁡(-‖x_i  -x_j ‖^2)
    2) return K
    ‘’’

# Define the Gaussian Process Regression function
def gaussian_regression(X_train, y_train, X_test):
    ‘’’
    8. 1)-2 (12 pts)
    YOUR WORK HERE
    Perform Gaussian Process Regression
    Given test data X_test, we compute μ^* and Σ^* using the following formula.
μ^*=k(x_test,x)K(x,x)^(-1) y
Σ^*=k(x_test,x_test )-k(x_test,x)K(x,x)^(-1) k(x,x_test )
    where
    x : X_train 
    y : y_train
    x_test : X_test
    μ^* : mean of X_test
    Σ^* : Covariance of X_test

    Suppose the size of vector x=6 and vector x_test=2, respectively. The shapes of matrices in above formula are as follows.
k(x,x_test ).shape=(6,2)
k(x_test,x).shape=(2,6)
k(x_test,x_test ).shape=(2,2)
K(x,x).shape=(6,6)
y.shape=(6,)

    You have to the following.
    1) Calculate the inverse of covariance matrix K(x,x)^(-1) of the training data x 
    2) Calculate the covariance k(x,x_test ) between training data x and test data x_test
    3) Calculate the covariance matrix k(x_test,x_test ) of the test points
    4) Compute μ^* the mean of the posterior predictive distribution
    5) Compute the covariance Σ^* of the posterior predictive distribution
    7) return μ^* and Σ^*
    ‘’’

# Suppose you do the following 6 experiments changing the values of ‘X_train’.
# ‘y_train’ is its corresponding error function value
X_train = np.array([[1], [2], [4], [6], [8], [9]])
y_train = true_function(X_train).ravel()

# Let’s estimate the error function value when x=2.2
X_test=np.array([[2.2]])

‘’’
8. 1)-3 (4 pts)
YOUR WORK HERE
1) By using ‘gaussian_regression’ function, estimate mean and covariance of ‘X_test’
2) Compute standard deviation(s.d.) of X_test values (diagonal value of covariance is variance value)
‘’’

2) [4 pts] Now X_test=np.array([[2.2]]) is changed to as follows.
X_test = np.linspace(0, 10, 10).reshape(-1, 1)
By using ‘gaussian_regression’, show the mean value vector μ^* and s.d. vector for X_test, respectively.


3) [4 pts] (Lower Confidence Bound) For each value in X_test of Q. 3), compute lower confidence bound value.
‘’’
8. 3) 
YOUR WORK HERE
1) For each X_test value, compute the following LCB value.
LCB(x)=μ(x)-κ⋅σ(x)
μ: mean
σ: standard deviation
2) Choose your next hyperparameter value X based on above results
‘’’

4) [extra 4 pts] After Q. 3) is finished, at the end of the program, run the following and show the Gaussian distribution of error function.

# Plot the results
# mean: mean value vector of X_test
# sd: standard deviation vector X_test
plt.figure()
plt.plot(X_test, true_function(X_test), 'r:', label="True function")
plt.plot(X_train, y_train, 'r.', markersize=10, label="Training data")
plt.plot(X_test, mean, 'b-', label="Prediction")
plt.fill_between(X_test.ravel(), mean - 1.96*sd, mean + 1.96*sd, alpha=0.2, color='b', label="Confidence interval")
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()
