import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import loguniform
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import corner

### DATA AND PARAMETERS ########################################################################################

data = np.load("../solutions/transient.npy")
data = data.T

N = np.shape(data)[1]   #number of data points
ndim = 4   #number of parameters

#prior intervals
t0min, t0max = 0, 100
Amin, Amax = 0,50
bmin, bmax = 0, 50
alphamin, alphamax = np.exp(-5), np.exp(5)
sigmaWmin, sigmaWmax = np.exp(-2), np.exp(2)

x = np.linspace(0, 100, 100)   #xgrid for plots

### plot the raw data ###
plt.figure(figsize=(12,4))
plt.scatter(data[0], data[1], color='black')
plt.errorbar(data[0], data[1], data[2], linestyle='None', ecolor='gainsboro', capsize=3)
plt.title("Burst data")
plt.xlabel("time")
plt.ylabel("flux")
plt.show()

#%%
### FUNCTIONS ##################################################################################################

def burst(theta,t):
    b, A, t0, alpha = theta
    return np.where(t<t0, b, b+A*np.exp(-alpha*(t-t0)))

def gprofile(theta,t):
    A, b, t0, sigmaW = theta     
    return b + A * np.exp(-( ((t-t0)/sigmaW)**2) / 2)

def loglike(theta, data, model):
    x, y, sigma_y = data
    if model =='burst':
        y_fit = burst(theta, x)
    elif model == 'gprofile':
        y_fit = gprofile(theta, x)
    return -0.5 * np.sum( (y-y_fit)**2 / sigma_y**2 ) 

def ptform(u,model):
    x = np.array(u)  #u is uniformely distributed between [0, 1]
    #I want to transform them in the form of the variable priors
    x[0] = uniform(loc=Amin, scale=Amax-Amin).ppf(u[0])
    x[1] = uniform(loc=bmin, scale=bmax-bmin).ppf(u[1])
    x[2] = uniform(loc=t0min, scale=t0max-t0min).ppf(u[2])
    if model =='burst':
        x[3] = loguniform.ppf(u[3], alphamin, alphamax)
    elif model =='gprofile':
        x[3] = loguniform.ppf(u[3], sigmaWmin, sigmaWmax)
    return x


#%%    
### NESTED SAMPLING ############################################################################################

### burst model ###
sampler = dynesty.NestedSampler(loglike, ptform, ndim, logl_args=[data,'burst'], ptform_args=['burst'],
                                nlive=250)   #decrease live points (less sampling efficiency, less run time)
sampler.run_nested()
sresults_b = sampler.results

#%%
### gaussian profile ###
print("\n")
sampler = dynesty.NestedSampler(loglike, ptform, ndim, logl_args=[data,'gprofile'], ptform_args=['gprofile'],
                                nlive=250)   #decrease live points (less sampling efficiency, less run time)
sampler.run_nested()
sresults_g = sampler.results


#%%
### PLOTS ######################################################################################################

### burst model ###
print("\n\n--- BURST MODEL ---")

rfig, raxes = dyplot.runplot(sresults_b)   #summary of the run

tfig, taxes = dyplot.traceplot(sresults_b)   #trace plots

#from weighted samples (dynesty output) to equally weighted samples
samples = sresults_b.samples
weights = np.exp(sresults_b.logwt - sresults_b.logz[-1])
burst_samples = dyfunc.resample_equal(samples, weights)

labels_b = ['b', 'A', 't0', 'alpha']
fig = corner.corner(burst_samples, labels=labels_b, color='navy')
fig.suptitle("Burst parameters");   #corner plot with equally weighted samples

evidence_b = np.exp(sresults_b.logz[-1])   #compute the evidence


### summary statitic ###
print("\nMedian and 90% credible region:")
for i in range(ndim):
    median = np.median(burst_samples[:,i])
    q1, q2 = np.percentile(burst_samples[:,i], [5, 95])
    print(labels_b[i], ":\t", np.round(median, 2), "\t+", np.round(q1, 2), "-", np.round(q2, 2))

print("")
sresults_b.summary()   #print dynesty summary


### plot results ###
plt.figure(figsize=(12,4))
plt.scatter(data[0], data[1], color='black')
plt.errorbar(data[0], data[1], data[2], linestyle='None', ecolor='gainsboro', capsize=3)

indices = np.random.randint(len(burst_samples), size=100)   #select 100 random indeces to plot
for i in indices:
    sample = burst_samples[i]
    plt.plot(x, burst(sample, x), "crimson", alpha=0.1)

plt.title("Burst model")
plt.xlabel("time")
plt.ylabel("flux")
plt.show()

#%%
### gaussian profile ###
print("\n\n--- GAUSSIAN PROFILE ---")

rfig, raxes = dyplot.runplot(sresults_g)   #summary of the run

tfig, taxes = dyplot.traceplot(sresults_g)   #trace plots

#from weighted samples (dynesty output) to equally weighted samples
samples = sresults_g.samples
weights = np.exp(sresults_g.logwt - sresults_g.logz[-1])
gprofile_samples = dyfunc.resample_equal(samples, weights)

labels_g = ['b', 'A', 't0', 'sigmaW']
fig = corner.corner(gprofile_samples, labels=labels_g, color='navy')
fig.suptitle("Gaussian profile parameters");   #corner plot with equally weighted samples

evidence_g = np.exp(sresults_g.logz[-1])   #compute the evidence


### summary statitic ###
print("\nMedian and 90% credible region:")
for i in range(ndim):
    median = np.median(gprofile_samples[:,i])
    q1, q2 = np.percentile(gprofile_samples[:,i], [5, 95])
    print(labels_g[i], ":\t", np.round(median, 2), "\t+", np.round(q1, 2), "-", np.round(q2, 2))

print("")
sresults_g.summary()   #print dynesty summary


### plot results ###
plt.figure(figsize=(12,4))
plt.scatter(data[0], data[1], color='black')
plt.errorbar(data[0], data[1], data[2], linestyle='None', ecolor='gainsboro', capsize=3)

indices = np.random.randint(len(gprofile_samples), size=100)   #select 100 random indeces to plot
for i in indices:
    sample = gprofile_samples[i]
    plt.plot(x, gprofile(sample, x), "limegreen", alpha=0.1)

plt.title("Gaussian profile")
plt.xlabel("time")
plt.ylabel("flux")
plt.show()


#%%
### MODEL SELECTION ############################################################################################

#compute the odds ratio factor
oddsr = evidence_b / evidence_g

print("\n\nODDS RATIO:", oddsr)
print("\nBetween 30 and 100 it is a VERY STRONG evidence in favour of the burst model")
print("Over 100 it is a DECISIVE evidence in favour of the burst model")