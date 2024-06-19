import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from astropy.cosmology import LambdaCDM
import emcee
import corner
from tqdm import tqdm
import warnings

################################################################################################################

ignore_warnings = True
if ignore_warnings:
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

### DATA #######################################################################################################

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)

z_sample_sk = z_sample[:, np.newaxis]

plt.figure()
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1, label='data')
plt.title("Raw data")
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
plt.show()

N = len(z_sample)

#for the MCMC
ndim = 2   #number of parameters
nwalkers = 10   #numebr of walkers 
nsteps = 10**4   #number of steps


#%%
### FUNCTIONS ##################################################################################################

def error(X, y, fit_method):
    Y = fit_method.predict(X)
    return np.sqrt( np.sum( (y-Y)**2 ) / len(X) )

def mu_LCDM(theta, z):
    H0, Om = theta
    model = LambdaCDM(H0=H0, Om0=Om, Ode0=1-Om)
    return model.distmod(z).value

def log_likelihood(theta, z):
    H0, Om = theta
    L = np.prod( np.exp( - (mu_sample - mu_LCDM(theta, z))**2 / (2 * dmu**2) ) )
    return np.log(L)

def log_prior(theta):
    H0, Om = theta
    if 50 < H0 < 100 and 0.05 < Om < 1 :
        return 0.0
    else:
        return -np.inf
    
def log_posterior(theta, z):
    if not np.isfinite(log_prior(theta)):
        return -np.inf
    return log_prior(theta) + log_likelihood(theta, z)


#%%
### CROSS VALIDATION OF RGB ####################################################################################

lenght_scales = np.linspace(0.1, 10, 20)

Xtrain, Xtest, ytrain, ytest, err_ytrain, err_ytest = train_test_split(z_sample, mu_sample, dmu, test_size=0.20)
Xtrain_sk = Xtrain[:, np.newaxis]
Xtest_sk = Xtest[:, np.newaxis]

TrainErr = []
CvErr = []
print("\n--- Cross validation for l ---")
for l in tqdm(lenght_scales):
    k = 1.0 * kernels.RBF(l, length_scale_bounds='fixed')
    gp = GaussianProcessRegressor(kernel=k, alpha=err_ytrain**2)
    gp.fit(Xtrain_sk, ytrain)
    TrainErr.append(error(Xtrain_sk, ytrain, gp))   #computing the training error
    CvErr.append(error(Xtest_sk, ytest, gp))   #computing the cross validation error

plt.figure()
plt.plot(lenght_scales, TrainErr, color='navy', label="Training err")
plt.plot(lenght_scales, CvErr, color='navy', ls='dashed', label="CV error")
plt.xlabel('l')
plt.xlim(10,0)
plt.title("Cross validation - without Kfold")
plt.legend()
plt.show()

#the cross validation error doesn't go up even for very large values of l


#%%
### K FOLDING OF RBF ###########################################################################################

lenght_scales = np.linspace(0.1, 10, 20)   #range of lenght scales to test
K = 20   #number of K-folds

TrainErr = []
CvErr = []
print("\n--- Kfold ---")
kf = KFold(n_splits=K, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(z_sample_sk)):
    print("Fold:", i)
    print("  Train:", train_index)
    print("  Test:", test_index)

    Xtrain = z_sample_sk[train_index]
    Xtest = z_sample_sk[test_index]
    ytrain = mu_sample[train_index]
    ytest = mu_sample[[test_index]]
    err_ytrain = dmu[train_index]
    err_ytest = dmu[test_index]

    te = []
    cve = []
    for l in tqdm(lenght_scales):
        k = 1.0 * kernels.RBF(l, length_scale_bounds='fixed')
        gp = GaussianProcessRegressor(kernel=k, alpha=err_ytrain**2)
        gp.fit(Xtrain, ytrain)
        te.append(error(Xtrain, ytrain, gp))   #computing the training error on this fold
        cve.append(error(Xtest, ytest, gp))   #computing the cross validation error on this fold
    
    TrainErr.append(np.array(te))
    CvErr.append(np.array(cve))

TrainErr_med = np.median(TrainErr, axis=0)
CvErr_med = np.median(CvErr, axis=0)

plt.figure()
plt.plot(lenght_scales, TrainErr_med, color='navy', label="Training err")
plt.plot(lenght_scales, CvErr_med, color='navy', ls='dashed', label="CV error")
plt.xlabel('l')
plt.xlim(10,0)
plt.title("Cross validation - Kfold")
plt.legend()
plt.show()


#%%
### GPR FIT ####################################################################################################

### RBF kernel ###
l = 10   #chosen lenght scale
k = 1.0 * kernels.RBF(l, length_scale_bounds='fixed')
gp = GaussianProcessRegressor(kernel=k, alpha=dmu**2)
gp.fit(z_sample_sk, mu_sample)

x = np.linspace(0, 2, 1000)
y_pred, dy_pred = gp.predict(x[:, np.newaxis], return_std=True)

plt.figure()
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.plot(x, y_pred, color='crimson', label='prediction')
plt.fill_between(x, y_pred - dy_pred, y_pred + dy_pred, color='crimson', alpha=0.3, label="$1\\sigma$")
plt.fill_between(x, y_pred - 2*dy_pred, y_pred + 2*dy_pred, color='crimson', alpha=0.15, label="$2\\sigma$")
plt.title("GPR fit")
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
plt.show()

#it's resonable to have bigger errors for very low z and high z, since we have less points
# which are also spread apart


#%%
### LAMBDA-CDM #################################################################################################

### sampling and MCMC ###
theta_guess = np.array([67, 0.3])   #starting guesses

print("\n--- MCMC ---")
start_guess = theta_guess + 10**(-1) * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[z_sample])
sampler.run_mcmc(start_guess, nsteps, progress=True);


### plotting the trace ###
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$H_0$", "$\\Omega_m$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], color="g", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number");


### burn in ###
tau = sampler.get_autocorr_time()
print("\n\nAutocorrelation time:", tau)

burn = 3 * int(np.max(tau))
thin = int(np.max(tau))
flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
print("After burning", burn, "values and thinning by", thin, "I obtain", flat_samples.shape, "samplings")

fig = corner.corner(flat_samples, labels=labels, levels=[0.68,0.95], color='g')
fig.suptitle("$\\Lambda CDM$ parameters");


### plot results ###
plt.figure()
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1, label='data')

indices = np.random.randint(len(flat_samples), size=100)
for i in indices:
    sample = flat_samples[i]
    plt.plot(x, mu_LCDM(sample, x), "g", alpha=0.1)

plt.title("Raw data")
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
plt.show()


#%%
### CLONING DATA ###############################################################################################

z_clone = np.linspace(0.0, 2.0, N*10)
mu_mu, mu_sigma = gp.predict(z_clone[:, np.newaxis], return_std=True)
mu_clone = np.random.normal(loc=mu_mu, scale=mu_sigma, size=N*10)

plt.figure()
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.scatter(z_clone, mu_clone, color='royalblue', marker='.', label='cloned data')
plt.title("Cloning data")
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
plt.show()


#%%
### LEARNING CURVE #############################################################################################

TrainErr = []
CvErr = []

step = 2
train = np.arange(6, N+step, step)
print("\n--- Learning curve ---")
ind = np.random.choice(np.arange(N), N, replace=False)
for n in tqdm(train):
    z_sample_cut = z_sample[ind[0:n]]
    mu_sample_cut = mu_sample[ind[0:n]]
    Xtrain, Xtest, ytrain, ytest = train_test_split(z_sample_cut, mu_sample_cut, test_size=0.20)
    Xtrain_sk = Xtrain[:, np.newaxis]
    Xtest_sk = Xtest[:, np.newaxis]
    
    k = 1.0 * kernels.RBF(l, length_scale_bounds='fixed')
    gp = GaussianProcessRegressor(kernel=k)
    gp.fit(Xtrain_sk, ytrain)
    TrainErr.append(error(Xtrain_sk, ytrain, gp))   #computing the training error
    CvErr.append(error(Xtest_sk, ytest, gp))   #computing the cross validation error

plt.figure()
plt.plot(train, TrainErr, color='navy', label="Training err")
plt.plot(train, CvErr, color='navy', ls='dashed', label="CV error")
plt.axvline(N, color='crimson', label="our dataset")
plt.xlabel('$N_{data}$')
plt.title("Learning curve")
plt.ylim(0, 3)
plt.legend()
plt.show()








