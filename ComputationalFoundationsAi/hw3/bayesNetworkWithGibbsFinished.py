
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
    p_c0 = P_C[0] * P_S_given_C[(0, s)] * P_R_given_C[(0, r)]
    p_c1 = P_C[1] * P_S_given_C[(1, s)] * P_R_given_C[(1, r)]
    norm = p_c0 + p_c1
    p_c1_given_sr = p_c1 / norm
    return 1 if random.random() < p_c1_given_sr else 0

def sample_S(c, r, w):
    # ‘’’  
    # Using gibbs sampling compute P(C=0|S,R) and P(C=1|S,R)
    # Use P(C|S,R) ∝ P(C) * P(S|C) * P(R|C)
    # return the value of C (0 or 1)
    # ‘’’
    p_s0 = P_S_given_C[(c, 0)] * P_W_given_S_R[(0, r, w)]
    p_s1 = P_S_given_C[(c, 1)] * P_W_given_S_R[(1, r, w)]
    norm = p_s0 + p_s1
    p_s1_given_crw = p_s1 / norm
    return 1 if random.random() < p_s1_given_crw else 0

def sample_R(c, s, w):
    # ‘’’
    # Using gibbs sampling compute P(C=0|S,R) and P(C=1|S,R)
    # Use P(C|S,R) ∝ P(C) * P(S|C) * P(R|C)
    # return the value of C (0 or 1)
    # ‘’’
    p_r0 = P_R_given_C[(c, 0)] * P_W_given_S_R[(s, 0, w)]
    p_r1 = P_R_given_C[(c, 1)] * P_W_given_S_R[(s, 1, w)]
    norm = p_r0 + p_r1
    p_r1_given_csw = p_r1 / norm
    return 1 if random.random() < p_r1_given_csw else 0

# Gibbs Sampling
# w=1 (evidence)
evidence_w = 1
iterations = 600

"""Perform Gibbs Sampling to estimate P(C | W=1)"""
samples = []

# Randomly initialize c,s,r (w is fixed as evidence)
c = random.randint(0, 1)
s = random.randint(0, 1)
r = random.randint(0, 1)

for i in range(iterations):
    # Sample c, s, and r in this order
    c = sample_C(s, r)
    s = sample_S(c, r, evidence_w)
    r = sample_R(c, s, evidence_w)

    samples.append(c)

# Compute P(C=1 | W=1)
p_c1_given_w1 = sum(samples) / len(samples)

print(f"Estimated P(C=1 | W=1) using Gibbs sampling: {p_c1_given_w1:.4f}")
