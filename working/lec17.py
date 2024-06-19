import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from astroML.linear_model import PolynomialRegression
from astroML.linear_model import BasisFunctionRegression
from astroML.linear_model import NadarayaWatson
from sklearn.model_selection import KFold

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

max_deg = 20   #complexity of the fits
x = np.linspace(0, 2, 1000)   #x axis

print_pol_coeff = False
print_foldings = False

#%%
### FUNCTIONS ##################################################################################################

def error(X, y, fit_method):
    Y = fit_method.predict(X)
    return np.sqrt( np.sum( (y-Y)**2 ) / len(X) )


#%%
### POLYNOMIAL REGRESSION ##########################################################################################

print("\n--- POLYNOMIAL REGRESSION ---")
print("    (blue plots)\n")

fig, axs = plt.subplots(4, 5, figsize=(24, 16), sharex=True, sharey=True)
axs = axs.flatten()
coeff_lr = []

for i in range(max_deg):
    
    model = PolynomialRegression(i+1)
    model.fit(z_sample_sk, mu_sample, dmu)   #fitting polinomials with different degrees
    coeff_lr.append(model.coef_)   #saving the coefficients
    
    mu_fitted = model.predict(x[:, np.newaxis])   #predict the values from the regression
    
    if print_pol_coeff:
        print(i+1, "degree polynomial:")
        print(coeff_lr[i], "\n")
    axs[i].errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)   #plot the data
    axs[i].plot(x, mu_fitted, color='dodgerblue')   #plot the polynomial fit on top
    axs[i].set_title("Fit with a " + str(i+1) + " degree polynomial")
    if i in [15, 16, 17, 18, 19]:
        axs[i].set_xlabel("z")
    if i in [0, 5, 10, 15]:
        axs[i].set_ylabel("$\mu$")
    axs[i].set_xlim(0,2)
    axs[i].set_ylim(35,50);


#%%
### BASIS FUNCTION REGRESSION ##################################################################################

print("\n--- BASIS FUNCTION REGRESSION ---")
print("    (red plots)\n")

fig, axs = plt.subplots(4, 5, figsize=(24, 16), sharex=True, sharey=True)
axs = axs.flatten()

for i in range(max_deg):
    
    grid = np.linspace(0, 1, i+2)[:, np.newaxis]   #defining the mean of the gaussians
    sigma = 2*(grid[1] - grid[0])   #defining the sigma of the gaussians (twice the distance between two mu)
    model = BasisFunctionRegression('gaussian', mu=grid, sigma=sigma)
    model.fit(z_sample_sk, mu_sample, dmu)   #fitting the data
    mu_fitted = model.predict(x[:,np.newaxis])   #predict the values from the regression
    
    axs[i].errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)   #plot the data
    axs[i].plot(x, mu_fitted, color='crimson')   #plot the fit on top
    axs[i].set_title("Fit with " + str(i+2) + " gaussians")
    if i in [15, 16, 17, 18, 19]:
        axs[i].set_xlabel("z")
    if i in [0, 5, 10, 15]:
        axs[i].set_ylabel("$\mu$")
    axs[i].set_xlim(0,2)
    axs[i].set_ylim(35,50);
    
    
#%%
### KERNEL REGRESSION ##########################################################################################

print("\n--- KERNEL REGRESSION ---")
print("    (yellow plots)\n")

fig, axs = plt.subplots(4, 5, figsize=(24, 16), sharex=True, sharey=True)
axs = axs.flatten()

for i in range(max_deg):
    
    widht = 1/(i+1)   #defining the bandwidht
    model = NadarayaWatson(kernel='gaussian', h=widht)
    model.fit(z_sample_sk, mu_sample, dmu)   #fitting the data
    mu_fitted = model.predict(x[:,np.newaxis])   #predict the values from the regression
    
    axs[i].errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)   #plot the data
    axs[i].plot(x, mu_fitted, color='orange')   #plot the fit on top
    axs[i].set_title("Fit with kernel bandwidht " + str(np.round(1/(i+1), 2)))
    if i in [15, 16, 17, 18, 19]:
        axs[i].set_xlabel("z")
    if i in [0, 5, 10, 15]:
        axs[i].set_ylabel("$\mu$")
    axs[i].set_xlim(0,2)
    axs[i].set_ylim(35,50);
    
    
#%%
### K FOLDING ##################################################################################################    
    
K = 15

TrainErr_lr = []
CvErr_lr = []
TrainErr_bf = []
CvErr_bf = []
TrainErr_ker = []
CvErr_ker = []

print("\n--- Kfold ---")
kf = KFold(n_splits=K, shuffle=True)
for l, (train_index, test_index) in enumerate(kf.split(z_sample_sk)):
    if print_foldings:
        print("Fold:", l+1)
        print("  Train:", train_index)
        print("  Test:", test_index)
        
    #deviding the data in foldings
    Xtrain = z_sample_sk[train_index]
    Xtest = z_sample_sk[test_index]
    ytrain = mu_sample[train_index]
    ytest = mu_sample[[test_index]]
    err_ytrain = dmu[train_index]
    err_ytest = dmu[test_index]
    
    ### polynomial regression ###
    te_lr = []
    cve_lr = []
    for i in range(max_deg):
        model = PolynomialRegression(i+1)
        model.fit(Xtrain, ytrain, err_ytrain)   #fitting the train set
        te_lr.append(error(Xtrain, ytrain, model))   #computing the training error on this fold
        cve_lr.append(error(Xtest, ytest, model))   #computing the cross validation error on this fold       
    TrainErr_lr.append(np.array(te_lr))
    CvErr_lr.append(np.array(cve_lr))
    
    ### basis function regression ###
    te_bf = []
    cve_bf = []
    for i in range(max_deg):
        grid = np.linspace(0, 1, i+2)[:, np.newaxis]   #defining the mean of the gaussians
        sigma = 2*(grid[1] - grid[0])   #defining the sigma of the gaussians (twice the distance between two mu)
        model = BasisFunctionRegression('gaussian', mu=grid, sigma=sigma)
        model.fit(Xtrain, ytrain, err_ytrain)   #fitting the train set
        te_bf.append(error(Xtrain, ytrain, model))   #computing the training error on this fold
        cve_bf.append(error(Xtest, ytest, model))   #computing the cross validation error on this fold      
    TrainErr_bf.append(np.array(te_bf))
    CvErr_bf.append(np.array(cve_bf))
    
    ### kernel regression ###
    te_ker = []
    cve_ker = []
    for i in range(max_deg):
        widht = 1/(i+1)   #defining the bandwidht
        model = NadarayaWatson(kernel='gaussian', h=widht)
        model.fit(Xtrain, ytrain, err_ytrain)   #fitting the train set
        te_ker.append(error(Xtrain, ytrain, model))   #computing the training error on this fold
        cve_ker.append(error(Xtest, ytest, model))   #computing the cross validation error on this fold       
    TrainErr_ker.append(np.array(te_ker))
    CvErr_ker.append(np.array(cve_ker))

TElr_med = np.median(TrainErr_lr, axis=0)
CElr_med = np.median(CvErr_lr, axis=0)
TEbf_med = np.median(TrainErr_bf, axis=0)
CEbf_med = np.median(CvErr_bf, axis=0)
TEker_med = np.median(TrainErr_ker, axis=0)
CEker_med = np.median(CvErr_ker, axis=0)

### plots ###
x_ax = np.arange(max_deg)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(x_ax+1, TElr_med, color='dodgerblue', label="Training err")
axs[0].plot(x_ax+1, CElr_med, color='dodgerblue', ls='dashed', label="CV error")
axs[1].plot(x_ax+2, TEbf_med, color='crimson', label="Training err")
axs[1].plot(x_ax+2, CEbf_med, color='crimson', ls='dashed', label="CV error")
axs[2].plot(1/(x_ax+1), TEker_med, color='orange', label="Training err")
axs[2].plot(1/(x_ax+1), CEker_med, color='orange', ls='dashed', label="CV error")
axs[0].set_title("Linear regression")
axs[1].set_title("Basis function regression")
axs[2].set_title("Kernel regression")
axs[0].set_ylabel("error")
axs[0].set_xlabel("degree of the polynomial")
axs[1].set_xlabel("number of gaussians")
axs[2].set_xlabel("bandwidht of the kernel")
axs[2].set_xlim(1.05, 0)
axs[0].set_xticks(np.arange(1, max_deg+1, 2))
axs[1].set_xticks(np.arange(2, max_deg+2, 2))
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()       
        
#the cross validation varies a lot depending on the random seed
#I decided to do a Kfold instead
#it's still very sensible on what ends up in the training and test set :(
        
