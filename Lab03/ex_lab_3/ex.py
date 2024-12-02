from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('S', 'O'), ('S', 'L'), ('L', 'M'), ('S', 'M')])

# Defining individual CPDs.
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])  # S=0 non-spam, S=1 spam
cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.9, 0.3],    # P(O=0 | S=0), P(O=0 | S=1)
                           [0.1, 0.7]],   # P(O=1 | S=0), P(O=1 | S=1)
                   evidence=['S'],
                   evidence_card=[2])

cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.7, 0.2],    # P(L=0 | S=0), P(L=0 | S=1)
                           [0.3, 0.8]],   # P(L=1 | S=0), P(L=1 | S=1)
                   evidence=['S'],
                   evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.8, 0.4, 0.5, 0.1],    # P(M=0 | S=0, L=0), P(M=0 | S=0, L=1), P(M=0 | S=1, L=0), P(M=0 | S=1, L=1)
                           [0.2, 0.6, 0.5, 0.9]],   # P(M=1 | S=0, L=0), P(M=1 | S=0, L=1), P(M=1 | S=1, L=0), P(M=1 | S=1, L=1)
                   evidence=['S', 'L'],
                   evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)

# Verifying the model
assert model.check_model()

# Performing exact inference using Variable Elimination
infer = VariableElimination(model)
result = infer.query(variables=['S'], evidence={'O': 1, 'L': 1, 'M': 1})
print(result)

print("Independencies:")
print(model.local_independencies(['S', 'O', 'L', 'M']))

# Drawing the Bayesian Network
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()