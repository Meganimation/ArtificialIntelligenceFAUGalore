import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# 1. Define the structure
# I (Infection/Disease) causes C (Clinical Symptom) and T (Test Result)
model = BayesianNetwork([('I', 'C'), ('I', 'T')])

# 2. Define the CPDs (Conditional Probability Distributions)
# P(I)
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.8], [0.2]])

# P(C | I) - Columns are I=0, I=1
cpd_c = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.7, 0.4],   # P(C=0 | I)
                           [0.3, 0.6]],  # P(C=1 | I)
                   evidence=['I'], evidence_card=[2])

# P(T | I) - Columns are I=0, I=1
cpd_t = TabularCPD(variable='T', variable_card=2, 
                   values=[[0.9, 0.2],   # P(T=0 | I)
                           [0.1, 0.8]],  # P(T=1 | I)
                   evidence=['I'], evidence_card=[2])

model.add_cpds(cpd_i, cpd_c, cpd_t)

# 3. Visualization
plt.figure(figsize=(6, 4))
pos = {'I': (0, 1), 'C': (-1, 0), 'T': (1, 0)} # Manual positioning for clarity

nx.draw(model, pos, with_labels=True, node_size=3000, 
        node_color="skyblue", font_size=15, font_weight="bold", 
        arrowsize=20, edge_color="gray")

plt.title("Bayesian Network: Disease Inference Model")
plt.show()

# Verification
print(f"Model is valid: {model.check_model()}")