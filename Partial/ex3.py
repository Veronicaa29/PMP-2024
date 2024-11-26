import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Date pentru experimentul 1
n1 = 10
k1 = 3
alpha_prior1 = 1
beta_prior1 = 1
alpha_post1 = alpha_prior1 + k1
beta_post1 = beta_prior1 + n1 - k1

# Date pentru experimentul 2
n2 = 10
k2 = 5
# a) Prior uniform
alpha_prior2a = 1
beta_prior2a = 1
alpha_post2a = alpha_prior2a + k2
beta_post2a = beta_prior2a + n2 - k2

# b) Prior din experimentul 1
alpha_prior2b = alpha_post1
beta_prior2b = beta_post1
alpha_post2b = alpha_prior2b + k2
beta_post2b = beta_prior2b + n2 - k2

theta = np.linspace(0, 1, 100)


plt.figure(figsize=(8, 6))
plt.plot(theta, beta.pdf(theta, alpha_post1, beta_post1), label='Beta(4,8)')
plt.title('Distribuția a posteriori după primul experiment')
plt.xlabel('θ')
plt.ylabel('Densitatea de probabilitate')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(theta, beta.pdf(theta, alpha_post2a, beta_post2a), label='Beta(6,6)')
plt.title('Distributia a posteriori cu prior uniform (al doilea experiment)')
plt.xlabel('θ')
plt.ylabel('Densitatea de probabilitate')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(theta, beta.pdf(theta, alpha_post2b, beta_post2b), label='Beta(9,13)')
plt.title('Distributia a posteriori cu prior informativ (al doilea experiment)')
plt.xlabel('θ')
plt.ylabel('Densitatea de probabilitate')
plt.legend()
plt.show()
