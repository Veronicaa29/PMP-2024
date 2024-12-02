# Import necessary libraries
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovNetwork
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pgmpy.inference import BeliefPropagation
from itertools import product

# Formula din cerinta 
def clique_energy(state):
    A1, A2, A3, A4, A5 = state
    energy = 0

    # phi (A4, A5, A2)
    energy += np.exp(i4 * A4 + i5 * A5 + i2 * A2)

    # phi (A1, A2)
    energy += np.exp(i1 * A1 + i2 * A2)

    # phi (A1, A3)
    energy += np.exp(i1 * A1 + i3 * A3)

    # phi (A4, A3)
    energy += np.exp(i4 * A4 + i3 * A3)

    return energy

model = MarkovNetwork()

# Add nodes for the variables A1, A2, A3, A4, A5
model.add_nodes_from(range(1, 6))

# Add edges based on the connections described
edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 4), (4, 5)]
model.add_edges_from(edges)

# Visualize the Markov Network using a different layout
plt.figure(figsize=(7, 5))
# Try using shell layout or circular layout
pos = nx.shell_layout(model)

# Draw the graph
nx.draw(model, pos, with_labels=True, node_size=2000, font_weight='bold', node_color='skyblue')
plt.title("Markov Network")
plt.axis('off')
plt.show()

# Convert to undirected graph to find cliques
undirected_graph = model.to_undirected()
cliques = list(nx.find_cliques(undirected_graph))
print("Cliques in the model:")
for clique in cliques:
    print(clique)

i1, i2, i3, i4, i5 = 1, 1, 1, 1, 1

phi_A4_A5_A2 = DiscreteFactor(variables=[4, 5, 2], cardinality=[2, 2, 2],
                              values=[np.exp(i4 * 0 + i5 * 0 + i2 * 0), np.exp(i4 * 0 + i5 * 0 + i2 * 1),
                                      np.exp(i4 * 0 + i5 * 1 + i2 * 0), np.exp(i4 * 0 + i5 * 1 + i2 * 1),
                                      np.exp(i4 * 1 + i5 * 0 + i2 * 0), np.exp(i4 * 1 + i5 * 0 + i2 * 1),
                                      np.exp(i4 * 1 + i5 * 1 + i2 * 0), np.exp(i4 * 1 + i5 * 1 + i2 * 1)])

phi_A1_A2 = DiscreteFactor(variables=[1, 2], cardinality=[2, 2],
                           values=[np.exp(i1 * 0 + i2 * 0), np.exp(i1 * 0 + i2 * 1),
                                   np.exp(i1 * 1 + i2 * 0), np.exp(i1 * 1 + i2 * 1)])

phi_A1_A3 = DiscreteFactor(variables=[1, 3], cardinality=[2, 2],
                           values=[np.exp(i1 * 0 + i3 * 0), np.exp(i1 * 0 + i3 * 1),
                                   np.exp(i1 * 1 + i3 * 0), np.exp(i1 * 1 + i3 * 1)])

phi_A4_A3 = DiscreteFactor(variables=[4, 3], cardinality=[2, 2],
                           values=[np.exp(i4 * 0 + i3 * 0), np.exp(i4 * 0 + i3 * 1),
                                   np.exp(i4 * 1 + i3 * 0), np.exp(i4 * 1 + i3 * 1)])

model.add_factors(phi_A1_A2, phi_A1_A3, phi_A4_A5_A2, phi_A4_A3)

inference = BeliefPropagation(model)

max_prob_state = inference.map_query(variables=[1, 2, 3, 4, 5])

variable_names = {1: 'A1', 2: 'A2', 3: 'A3', 4: 'A4', 5: 'A5'}

# Print maximum probability states with variable names
max_prob_state_named = {variable_names[k]: v for k, v in max_prob_state.items()}
print("Stările de probabilitate maximă:", max_prob_state_named)

joint_probabilities = {}
z = 0

# Combinarea tuturor valorilor posibile (0, 1) pentru cele 5 variabile
for state in product([0, 1], repeat=5):
    energy = clique_energy(state)
    joint_probabilities[state] = energy
    z += energy

for state in joint_probabilities:
    joint_probabilities[state] /= z

print("Probabilitățile comune ale variabilelor:")
for state, prob in joint_probabilities.items():
    print(f"Starea {state}: probabilitate {prob}")
