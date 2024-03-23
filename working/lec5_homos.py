import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

################################################################################################################

# gaussian measurement
N = 7
mu_real = 1
sigma_real = 0.2

#%%
### PART 1 #####################################################################################################

### generating gaussian measurements ###
r = norm(loc=mu_real, scale=sigma_real).rvs(N)   #sample gaussian distributed

plt.figure()
plt.hist(r, bins=15)
plt.title("Measurements data")
plt.xlabel("r")
plt.show()


### likelihood computation ###
xgrid = np.linspace(0, 2, 1000)
L_r = []

plt.figure()
for m in tqdm(r) :
    temp = np.array(norm(loc=m, scale=sigma_real).pdf(xgrid))
    plt.plot(xgrid, temp)
    L_r.append(temp)
    
L = np.prod(L_r, axis=0)

max_L = xgrid[np.argmax(L)]   #find the max of the likelihood

plt.plot(xgrid, L, color='black', lw=2)
plt.axvline(max_L, color='black', linestyle='dashed')
plt.axvline(mu_real, color='black')
plt.title("Likelihood")
plt.xlabel("$\mu$(r)")
plt.ylabel("$p~(x_i~|~\mu,\sigma)$")
plt.show()


### compute the MLE ###
mean_r = np.mean(r)

print("\nCOMPARISON")
print("Likelihood maximum: ", max_L)
print("MLE (sample mean): ", mean_r)

#%%
### PART 2 #####################################################################################################

sigma_mu = np.diff(np.log(L), n=2)
sigma_mu /= (xgrid[1] - xgrid[0])**2
sigma_mu *= -1
sigma_mu = sigma_mu[np.argmax(L)]**(-0.5)

print("\n")
print("Fisher matrix error: ", sigma_mu)
print("Sigma of the mean: ", sigma_real/np.sqrt(N))

MLE_gaussian = norm(loc=mean_r, scale=sigma_real/np.sqrt(N)).pdf(xgrid)

C = np.max(MLE_gaussian)/np.max(L)

plt.figure()
plt.plot(xgrid, MLE_gaussian, color='deepskyblue', lw=2, label="Gaussian with MLE")
plt.plot(xgrid, C*L, color='black', linestyle='dashed', label="Likelihood")
plt.title("Guassians comparison")
plt.xlabel("$\mu$(r)")
plt.ylabel("$p~(x_i~|~\mu,\sigma)$")
plt.legend()
plt.show()
