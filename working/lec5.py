import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

################################################################################################################

# gaussian measurement
N = 5
mu_real = 1
sigma_real = 0.2

#%%
### PART 1 #####################################################################################################

### generating gaussian measurements ###
distG = norm(loc=mu_real, scale=sigma_real)
x = distG.rvs(N)

plt.figure()
plt.hist(x, bins=15)
plt.title("Measurements data")
plt.xlabel("x")
plt.show()


### likelihood computation ###
mu_proposed = np.linspace(0, 2, 1000)   #range of possible means to evaluate
L_x = []

plt.figure()
for m in tqdm(mu_proposed):
    temp = norm(loc=m, scale=sigma_real)   #likelihoods are gaussians
    L_temp = np.prod(temp.pdf(x))   #L as product of single likelihoods
    plt.plot(mu_proposed, temp.pdf(x))
    L_x.append(L_temp)

max_L = mu_proposed[np.argmax(L_x)]   #find the max of the likelihood


plt.plot(mu_proposed, L_x, color='black', lw=2)
plt.axvline(max_L, color='red')
plt.axvline(mu_real, color='black', linestyle='dashed')
plt.title("Likelihood logaritm")
plt.xlabel("$\mu$(x)")
plt.xlim(0.5, 1.5)
plt.show()


### compute the MLE ###
mean_x = np.mean(x)

print("\nCOMPARISON")
print("Likelihood maximum: ", max_L)
print("MLE (sample mean): ", mean_x)

#%%
### PART 2 #####################################################################################################
