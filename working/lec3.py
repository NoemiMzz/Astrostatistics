import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

################################################################################################################

N = 1000   #number of samples
s = 5   #sigma parameter
it = 5000   #iterations for fixed N

sample_hist = False

### FUNCTIONS ##################################################################################################

def f(x, s) :
    return x**3 * s * np.sqrt(2*np.pi)

#%%
### FIX N ######################################################################################################

result = 2*s**4   #expexted result


### Montecarlo integration ###
integral_fix = []   #stores the results for every interaction, fixing the number of samples N

for i in tqdm(range(it)) :

    #we can rewrite the integrand as a gaussian with mu=0 p(x,s) multiplied by f(x,s) defined above
    distG_sample = norm(loc=0 , scale=s)   #function p(x)
    
    x = abs(distG_sample.rvs(N))   #sample generation
    
    if sample_hist :
        plt.hist(f(x, s), density=True)
    
    #the 0.5 is due to the bounds of integration (they are [0, inf), while in a gaussian they are (-inf, inf))
    integral_fix.append(0.5 * np.mean(f(x, s)))   #integral computation (for every sample)


### results distribution ###
plt.figure()
plt.hist(integral_fix, density=True, bins=20, color='deepskyblue', alpha=0.5, label='Monte Carlo integration')

mu = np.mean(integral_fix)   #mu as the average
sigma = np.std(integral_fix)   #sigma as standard deviation

print("\nExpected result: ", result)
print("Mean of Montecarlo integrations: ", round(mu, 2))

distG = norm(loc=mu , scale=sigma)   #fit "by hand" of a gaussian

xgrid = np.linspace(result-3*sigma, result+3*sigma, 1000)
gauss = distG.pdf(xgrid)

plt.plot(xgrid, gauss, color='navy', label='fit by hand')
plt.title("Results distribution")
plt.xlabel("result")
plt.ylabel("frequency")
plt.legend()
plt.show()

#%%
### VARIABLE N #################################################################################################

integral_var = []   #stores the results varying the number of samples N
N_values = np.linspace(100, 15000, 100, dtype=int)

for n in tqdm(range(len(N_values))) :

    distG_sample = norm(loc=0 , scale=s)   #function p(x)
    
    x = abs(distG_sample.rvs(N_values[n]))   #sample generation
    
    if sample_hist :
        plt.hist(f(x, s), density=True)
    
    integral_var.append(0.5 * np.mean(f(x, s)))   #integral computation (for every N)

plt.figure()    
plt.plot(N_values, abs(np.array(integral_var) - result)/result , color='deepskyblue', alpha=0.5, label='% error')
plt.plot(N_values, N_values**(-1/2), color='crimson', label='expected $1/\\sqrt{N}$')
plt.title("Error trend increasing sample")
plt.ylim(10**(-2.5), 0.2)
plt.xlabel("N")
plt.ylabel("% error")
plt.legend()
plt.loglog()
plt.show()