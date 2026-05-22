
# Bayes Network –Gibbs sampling

import numpy as np
import random

# Define the conditional probability tables (CPTs)
P_C = {0: 0.5, 1: 0.5}

P_S_given_C = {
    (0, 0): 0.5, (0, 1): 0.5,
    (1, 0): 0.1, (1, 1): 0.9
}

P_R_given_C = {
    (0, 0): 0.8, (0, 1): 0.2,
    (1, 0): 0.2, (1, 1): 0.8
}

P_W_given_S_R = {
    (0, 0, 0): 1.0,  (0, 0, 1): 0.0,
    (0, 1, 0): 0.1,  (0, 1, 1): 0.9,
    (1, 0, 0): 0.1,  (1, 0, 1): 0.9,
    (1, 1, 0): 0.01, (1, 1, 1): 0.99
}

# Sampling function for a variable given Markov blanket
def sample_C(s, r):
    # ‘’’  
    # Using gibbs sampling compute P(C=0|S,R) and P(C=1|S,R)
    # Use P(C|S,R) ∝ P(C) * P(S|C) * P(R|C)
    # return the value of C (0 or 1)
    # ‘’’

def sample_S(c, r, w):
    # ‘’’  
    # Using gibbs sampling compute P(C=0|S,R) and P(C=1|S,R)
    # Use P(C|S,R) ∝ P(C) * P(S|C) * P(R|C)
    # return the value of C (0 or 1)
    # ‘’’

def sample_R(c, s, w):
    # ‘’’
    # Using gibbs sampling compute P(C=0|S,R) and P(C=1|S,R)
    # Use P(C|S,R) ∝ P(C) * P(S|C) * P(R|C)
    # return the value of C (0 or 1)
    # ‘’’

# # Gibbs Sampling
# # w=1 (evidence)
# evidence_w=1
# iterations=600

# """ Perform Gibbs Sampling to estimate P(C | W=1) """
# samples = []
    
# # Initialize randomly
# ‘’’


# Randomly initialize c,s,r, w=1 (0 or 1)
# ‘’’
    
# for i in range(iterations):
#     ‘’’
    
    
#     Sample c, s, and r in this order
#     Append the value of c into samples
#     ‘’’

#     samples.append(c)

# # Compute P(C=1 | W=1)
# ‘’’


# Compute P(C=1 | W=1) and print
