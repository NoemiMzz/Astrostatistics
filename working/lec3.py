import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

################################################################################################################

N = 1000   #number of samples
s = 5   #sigma parameter
it = 5000   #iterations for fix N

sample_hist = False

### FUNCTIONS ##################################################################################################

def f(x, s) :
    return x**3 * s * np.sqrt(2*np.pi)

#%%
### FIX N ######################################################################################################

result = 2*s**4


### Montecarlo integration ###
integral_fix = []

for i in tqdm(range(it)) :

    distG_sample = norm(loc=0 , scale=s)   #function p(x)
    
    x = abs(distG_sample.rvs(N))   #sample generation
    
    if sample_hist :
        plt.hist(f(x, s), density=True)
    
    integral_fix.append(0.5 * np.mean(f(x, s)))   #integral computation (for every sample)


### results distribution ###
plt.figure()
plt.hist(integral_fix, density=True, bins=20, color="deepskyblue")   #results

mu = np.mean(integral_fix)   #mu as the average
sigma = np.std(integral_fix)   #sigma as standard deviation

print("\nExpected result: ", result)
print("Mean of Montecarlo integrations: ", round(mu, 3), "\n")

distG = norm(loc=mu , scale=sigma)   #fit "by hand" of a gaussian

xgrid = np.linspace(result-3*sigma, result+3*sigma, 1000)
gauss = distG.pdf(xgrid)

plt.plot(xgrid, gauss, color="orange")
plt.title("Results distribution")
plt.xlabel("result")
plt.ylabel("frequency")
plt.show()

#%%
### VARIABLE N #################################################################################################

integral_var = []
N_values = np.linspace(100, N, int(N/100), dtype=int)

for n in tqdm(range(len(N_values))) :

    distG_sample = norm(loc=0 , scale=s)   #function p(x)
    
    x = abs(distG_sample.rvs(N_values[n]))   #sample generation
    
    if sample_hist :
        plt.hist(f(x, s), density=True)
    
    integral_var.append(0.5 * np.mean(f(x, s)))   #integral computation (for every N)

plt.figure()    
plt.plot(N_values, (np.array(integral_var) - result)/result , color="royalblue")
plt.title("Error increasing sample")
plt.xlabel("N")
plt.ylabel("% error")
plt.show()