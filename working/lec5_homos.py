import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

################################################################################################################

# gaussian measurement
N = 7
mu_real = 1
sigma_real = 0.2

#%%
### PART 1 #####################################################################################################

### generating gaussian measurements ###
r = norm(loc=mu_real, scale=sigma_real).rvs(N)   #sample gaussian distributed


### likelihood computation ###
xgrid = np.linspace(0, 2, 1000)
L_r = []

plt.figure()
for m in r :
    gauss = norm(loc=m, scale=sigma_real)   #plot the gaussian of every single measurement
    plt.plot(xgrid, gauss.pdf(xgrid), color='gainsboro')
    L_r.append(gauss.pdf(xgrid))
    
L = np.prod(L_r, axis=0)   #compute the likelihood as their product

max_L = xgrid[np.argmax(L)]   #find the max of the likelihood

plt.plot(xgrid, L, color='black', lw=2)
plt.axvline(max_L, color='black', linestyle='dashed')
plt.axvline(mu_real, color='crimson')
plt.title("Likelihood")
plt.xlabel("$\mu$(r)")
plt.ylabel("$p~(x_i~|~\mu,\sigma)$")
plt.show()


### compute the MLE ###
mean_r = np.mean(r)

print("\n-- COMPARISON --")
print("\nLikelihood maximum: ", max_L)
print("MLE (sample mean): ", mean_r)

#%%
### PART 2 #####################################################################################################

### compute the Fisher matrix ###
sigma_mu = np.diff(np.log(L), n=2)   #2nd order expansion
sigma_mu /= (xgrid[1] - xgrid[0])**2   #divide by mu interval
sigma_mu *= -1
sigma_mu = sigma_mu[np.argmax(L)]**(-0.5)   #inverse, square root and evaluation

### compute the MLE uncertanty ###
sigma_mean = sigma_real/np.sqrt(N)

print("\nFisher matrix error: ", sigma_mu)
print("Sigma of the mean: ", sigma_mean)

MLE_gaussian = norm(loc=mean_r, scale=sigma_mean).pdf(xgrid)

C = np.max(MLE_gaussian)/np.max(L)

plt.figure()
plt.plot(xgrid, MLE_gaussian, color='deepskyblue', lw=2, label="Gaussian with MLE")
plt.plot(xgrid, C*L, color='black', linestyle='dashed', label="Likelihood")
plt.title("Guassians comparison")
plt.xlabel("$\mu$(r)")
plt.ylabel("$p~(x_i~|~\mu,\sigma)$")
plt.legend()
plt.show()
