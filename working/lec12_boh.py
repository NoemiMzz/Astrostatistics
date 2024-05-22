import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import loguniform
import dynesty
import corner

### DATA AND PARAMETERS ########################################################################################

t = np.load("../solutions/transient.npy")[:, 0]
f = np.load("../solutions/transient.npy")[:, 1]
sigma = np.load("../solutions/transient.npy")[:, 2]

N = len(t)   #number of data points
ndim = 4   #number of parameters

### FUNCTIONS ##################################################################################################

def burst(theta):
    b, A, t0, alpha = theta
    print(type(t0))
    flux = []
    for time in t:
        if time < t0:
            flux.append(b)
        if time >= t0:
            flux.append( b + A * np.exp(- alpha * (time - t0)) )
    return np.array(flux)
    
def log_prior(u):
    p = np.zeros((4, len(u)))
    p[0] = uniform(0, 50).ppf(u[0])   #prior_b
    p[1] = uniform(0, 50).ppf(u[1])   #prior_A
    p[2] = uniform(0, 100).ppf(u[2])   #prior_t0
    p[3] = loguniform(-5, 5).ppf(u[3])   #prior_alpha
    return p
    
def log_likelihood(theta):
    b, A, t0, alpha = theta
    f_model = burst(theta)
    L = np.prod( np.exp( - (f - f_model)**2 / (2 * sigma**2) ) )
    return np.log(L)
    
################################################################################################################
#%%
### plot the raw data ###
plt.figure(figsize=(12,4))
plt.scatter(t, f, color='black')
plt.errorbar(t, f, sigma, linestyle='None', ecolor='gainsboro', capsize=3)
plt.title("Burst data")
plt.xlabel("time")
plt.ylabel("flux")
plt.show()


### starting guesses ###
t0_quick=50
A_quick=5
b_quick=10
alpha_quick=0.1

theta_quick= np.array([b_quick,A_quick,t0_quick,alpha_quick])   #from Gerosa's file

x = np.linspace(0, 100, 100)
plt.figure(figsize=(12,4))
plt.scatter(t, f, color='black')
plt.errorbar(t, f, sigma, linestyle='None', ecolor='gainsboro', capsize=3)
plt.plot(t, burst(theta_quick), color='limegreen')
plt.title("Burst data")
plt.xlabel("time")
plt.ylabel("flux")
plt.show()


#%%
### nested sampling ###
sampler = dynesty.NestedSampler(log_likelihood, log_prior, ndim)
#sampler.run_nested()
#sresults = sampler.results




