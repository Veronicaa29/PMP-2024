from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dimensiunea gridului
dimensiune_grid = (10, 10)

# Lista de culori predefinite
culori = [
    "red", "blue", "green", "yellow",
    "purple", "orange", "pink", "cyan",
    "brown", "lime"
]
stari = [
    "Pe loc", "Stanga", "Dreapta", "Sus", "Jos"
]
n_stari = len(stari)
start_probability = np.array([25/100, 18.75/100, 18.75/100, 18.75/100, 18.75/100])

# Citirea gridului
df = pd.read_csv('grid_culori.csv', header=None)
grid_culori = df.to_numpy

# Generarea secvenței de culori observate
observatii = ["red", "red", "lime", "yellow", "blue"]

# Mapare culori -> indecși
culoare_to_idx = {culoare: idx for idx, culoare in enumerate(culori)}
idx_to_culoare = {idx: culoare for culoare, idx in culoare_to_idx.items()}

# Transformăm secvența de observații în indecși
observatii_idx = [culoare_to_idx[c] for c in observatii]
# print(observatii_idx)

# Definim stările ascunse ca fiind toate pozițiile din grid (100 de stări)
numar_stari = dimensiune_grid[0] * dimensiune_grid[1]
stari_ascunse = [(i, j) for i in range(dimensiune_grid[0]) for j in range(dimensiune_grid[1])]
stare_to_idx = {stare: idx for idx, stare in enumerate(stari_ascunse)}
idx_to_stare = {idx: stare for stare, idx in stare_to_idx.items()}
# print(stare_to_idx)
# print(idx_to_stare)

# Matrice de tranziție
transitions = np.zeros((numar_stari, numar_stari))
n = 0
for i, j in stari_ascunse:
    vecini = [
        (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)  # sus, jos, stânga, dreapta
    ]
    vecini_valizi = [stare_to_idx[(x, y)] for x, y in vecini if 0 <= x < 10 and 0 <= y < 10]
    # print("___", vecini_valizi)
    ######
    new_transitions = np.zeros((numar_stari, numar_stari))
    for k in range(len(transitions[n])):
        if k in vecini_valizi:
            new_transitions[n][k] = 18.75/100

    print(new_transitions[n])
    n = n+1

transition_probability = np.array([
    new_transitions
])
# print(transition_probability)

# Matrice de emisie
emissions = np.zeros((numar_stari, len(culori)))
emission_probability = np.array([
    emissions
])

# Modelul HMM
model = hmm.CategoricalHMM(n_components=n_stari)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# Rulăm algoritmul Viterbi pentru secvența de observații
n_states = len(stari)
n_observations = len(observatii)
viterbi = np.zeros((n_states, n_observations))
backpointer = np.zeros((n_states, n_observations), dtype=int)

for t in range(1, n_observations):
    current_obs = observatii_idx[observatii_idx[t]]
    for s in range(n_states):
        max_prob = -1
        max_state = 0
        for s_prev in range(n_states):
            prob = viterbi[s_prev, t-1] * transition_probability[s_prev, s] * emission_probability[s, current_obs]
            if prob > max_prob:
                max_prob = prob
                max_state = s_prev
        viterbi[s, t] = max_prob
        backpointer[s, t] = max_state

best_path_prob = np.max(viterbi[:, -1])
best_last_state = np.argmax(viterbi[:, -1])

best_path = [best_last_state]
for t in range(n_observations-1, 0, -1):
    best_path.insert(0, backpointer[best_path[0], t])

secventa_stari = [stari[state] for state in best_path]

# Convertim secvența de stări în poziții din grid
drum = [idx_to_stare[idx] for idx in secventa_stari]

# Vizualizăm drumul pe grid
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(dimensiune_grid[0]):
    for j in range(dimensiune_grid[1]):
        culoare = grid_culori[i, j]
        ax.add_patch(plt.Rectangle((j, dimensiune_grid[0] - i - 1), 1, 1, color=culoare))
        ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, culoare,
                color="white", ha="center", va="center", fontsize=8, fontweight="bold")

# Evidențiem drumul rezultat
for idx, (i, j) in enumerate(drum):
    ax.add_patch(plt.Circle((j + 0.5, dimensiune_grid[0] - i - 0.5), 0.3, color="black", alpha=0.7))
    ax.text(j + 0.5, dimensiune_grid[0] - i - 0.5, str(idx + 1),
            color="white", ha="center", va="center", fontsize=10, fontweight="bold")

# Setări axă
ax.set_xlim(0, dimensiune_grid[1])
ax.set_ylim(0, dimensiune_grid[0])
ax.set_xticks(range(dimensiune_grid[1]))
ax.set_yticks(range(dimensiune_grid[0]))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(visible=True, color="black", linewidth=0.5)
ax.set_aspect("equal")
plt.title("Drumul rezultat al stărilor ascunse", fontsize=14)
plt.show()