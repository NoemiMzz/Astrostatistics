import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

### DATA AND PARAMETERS ########################################################################################

t = np.load("../solutions/transient.npy")[:, 0]
f = np.load("../solutions/transient.npy")[:, 1]
sigma = np.load("../solutions/transient.npy")[:, 2]

N = len(t)   #number of data points
ndim = 4   #number of parameters
nwalkers = 15   #numebr of walkers 
nsteps = 10**4   #number of steps

### FUNCTIONS ##################################################################################################

def burst(theta, t):
    b, A, t0, alpha = theta
    flux = []
    for time in t:
        if time < t0:
            flux.append(b)
        if time >= t0:
            flux.append( b + A * np.exp(- alpha * (time - t0)) )
    return np.array(flux)
    
def log_prior(theta):
    b, A, t0, alpha = theta
    if 0 < b < 50 and 0 < A < 50 and 0 < t0 < 100 and np.exp(-5) < alpha < np.exp(5):
        return 0.0
    else:
        return -np.inf
    
def log_likelihood(theta, t, f, sigma):
    b, A, t0, alpha = theta
    L = np.prod( np.exp( - (f - burst(theta, t))**2 / (2 * sigma**2) ) )
    return np.log(L)

def log_posterior(theta, t, f, sigma):
    if not np.isfinite(log_prior(theta)):
        return -np.inf
    return log_prior(theta) + log_likelihood(theta, t, f, sigma)

 
#%%   
################################################################################################################

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

theta_quick= np.array([b_quick, A_quick, t0_quick, alpha_quick])   #from Gerosa's file

x = np.linspace(0, 100, 100)
plt.figure(figsize=(12,4))
plt.scatter(t, f, color='black')
plt.errorbar(t, f, sigma, linestyle='None', ecolor='gainsboro', capsize=3)
plt.plot(x, burst(theta_quick, x), color='limegreen')
plt.title("Start guesses")
plt.xlabel("time")
plt.ylabel("flux")
plt.show()


#%%   
### MCMC #######################################################################################################

print("--- MCMC ---")
### sampling and MCMC ###
start_guess = theta_quick + 10**(-1) * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[t, f, sigma])   #set the MCMC
sampler.run_mcmc(start_guess, nsteps, progress=True);   #run the MCMC


### plotting the trace ###
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()   #get trace from sampler
labels = ["b", "A", "$t_0$", "$\\alpha$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], color="crimson", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number");


### burn in ###
tau = sampler.get_autocorr_time()
print("\n\nAutocorrelation time:", tau)

burn = 3 * int(np.max(tau))   #I don't use the first samples
thin = int(np.max(tau))   #I take one sample every tau, to get semi-indipendent samples
flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
print("\nAfter burning", burn, "values over", nsteps, "and thinning by", thin, "I obtain", flat_samples.shape[0], "samplings")

fig = corner.corner(flat_samples, labels=labels, levels=[0.68,0.95], color='crimson')
fig.suptitle("Burst parameters");


### plot results ###
plt.figure(figsize=(12,4))
plt.scatter(t, f, color='black')
plt.errorbar(t, f, sigma, linestyle='None', ecolor='gainsboro', capsize=3)

indices = np.random.randint(len(flat_samples), size=100)   #select 100 random indeces to plot
for i in indices:
    sample = flat_samples[i]
    plt.plot(x, burst(sample, x), "crimson", alpha=0.1)

plt.title("Burst data")
plt.xlabel("time")
plt.ylabel("flux")
plt.show()

print("\nMedian and 90% credible region:")
lab_print = ["b", "A", "t0", "alpha"]
### summary statitic ###
for i in range(ndim):
    median = np.median(flat_samples[:,i])
    q1, q2 = np.percentile(flat_samples[:,i], [5, 95])
    print(lab_print[i], ":\t", np.round(median, 2), "\t+", np.round(q1, 2), "-", np.round(q2, 2))
    



