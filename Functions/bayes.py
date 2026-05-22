from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('B', 'A'), ('E', 'A'), ('A', 'J'), ('A', 'M')])

cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.95], [0.05]]) # 0: False, 1: True
cpd_e = TabularCPD(variable='E', variable_card=2, values=[[0.9], [0.1]])
cpd_a = TabularCPD(variable='A', variable_card=2, 
                   values=[[0.95, 0.5, 0.15, 0.05],   # P(~A)
                           [0.05, 0.5, 0.85, 0.95]],  # P(A)
                   evidence=['B', 'E'], evidence_card=[2, 2])

cpd_j = TabularCPD(variable='J', variable_card=2,
                   values=[[0.95, 0.3],   # P(~J)
                           [0.05, 0.7]],  # P(J)
                   evidence=['A'], evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.85, 0.2],   # P(~M)
                           [0.15, 0.8]],  # P(M)
                   evidence=['A'], evidence_card=[2])

model.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m)
assert model.check_model()


infer = VariableElimination(model)
result = infer.query(variables=['E', 'J', 'M'], evidence={})
joint_prob = result.values[1, 0, 1] 

print(f"P(E=True, J=False, M=True) = {joint_prob:.9f}")