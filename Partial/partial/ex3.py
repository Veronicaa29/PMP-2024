import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# 0 - s, 1 - b
observatii = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 1])

with pm.Model() as model:
    # Priori pentru μ și σ
    p = pm.Beta("p", alpha=1, beta=1, shape=10)

    # Distribuția pentru observații
    obs = pm.Binomial("obs", p=p, observed=observatii)

    # Eșantionare
    trace = pm.sample(10, tune=5, random_seed=42)

with model:
    # pm.sample creeaza esantioane aleatorii din distributia a posteriori a parametrilor modelului
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Afisare sumar a posteriori (proprietatile distributiei)
print(az.summary(trace, var_names=["p"]))

# Vizualizare a distributiei a posteriori pentru fiecare p_i
az.plot_posterior(trace, var_names=["p"], hdi_prob=0.95)
