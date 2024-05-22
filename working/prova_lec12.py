import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit
from scipy import optimize
import corner
from IPython.display import display, Math
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import statsmodels.api as sm
#%%
data = np.load('Astrostatistics/solutions/transient.npy')
time = data[:,0]
flux = data[:,1]
flux_err = data[:,2]
#%%
# Plot the results
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.errorbar(time, flux, flux_err, 
            fmt='.k', lw=1, ecolor='gray');
#%%
# coding the model
def model(t, theta):
    A, b, t_0, alpha = theta
    
    arg = -alpha * (t - t_0)

    exp_term = np.exp(np.clip(arg, a_min=-700, a_max=700))
    
    return np.where(t < t_0, b, b + A * exp_term)
    
def model_fit(t, A, b, t_0, alpha):
    
    return np.where(t<t_0,b,b+A*np.exp(-alpha*(t-t_0)))
#%%
popt, pcov = curve_fit(model_fit, time, flux, p0=[5,10,50,0.1])
A_opt, b_opt, t0_opt, alpha_opt = popt
x_model = np.linspace(min(time), max(time), 100)
y_model = model_fit(x_model, A_opt, b_opt, t0_opt, alpha_opt)


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.errorbar(time, flux, flux_err, 
            fmt='.k', lw=1, ecolor='gray');

plt.plot(x_model, y_model, color='r')
plt.show()
#%%
A = popt[0]
b = popt[1]
t_0 = popt[2]
alpha = popt[3]
#%%
def loglike(theta):
    
    A, b, t_0, alpha = theta
    model_flux = model(time, theta)
    
    return -0.5 * np.sum(((flux - model_flux) / flux_err) ** 2)

def ptform(u):
    
    x = np.array(u)
    
    lower_bound = [0, 0, 0, np.exp(-5)]
    upper_bound = [50, 50, 100, np.exp(5)]
    
    x = [lower_bound[i] + u[i] * (upper_bound[i] - lower_bound[i]) for i in range(len(u))]
    
    return x
#%%
ndim = 4
sampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim, sample='rwalk')
sampler.run_nested()
sresults = sampler.results
#%%
