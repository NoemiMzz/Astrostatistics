import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from numpy.polynomial.polynomial import Polynomial
from astroML.linear_model import LinearRegression
from astroML.linear_model import PolynomialRegression
from astroML.linear_model import BasisFunctionRegression
from astroML.linear_model import NadarayaWatson
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm

### DATA #######################################################################################################

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)

z_sample_sk = z_sample[:, np.newaxis]

plt.figure()
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.title("Raw data")
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
plt.show()

max_deg = 20

print_pol_coeff = False
print_foldings = True

#%%
### FUNCTIONS ##################################################################################################

def error(X, y, fit_method):
    Y = fit_method.predict(X)
    return np.sqrt( np.sum( (y-Y)**2 ) / len(X) )

#%%
### LEAST SQUARES FIT ##########################################################################################

coeff_ls = []
for i in range(max_deg):
    model = Polynomial.fit(z_sample, mu_sample, i+1, w=1/dmu)   #fitting polinomials with different degrees
    coeff_ls.append(model.convert().coef)

print("\n--- LEAST SQUARES FIT ---")
print("    (green plots)\n")
x = np.linspace(0, 2, 1000)
fig, axs = plt.subplots(4, 5, figsize=(25, 20), sharex=True, sharey=True)
axs = axs.flatten()
for i in range(max_deg):
    if print_pol_coeff:
        print(i+1, "degree polynomial:")
        print(coeff_ls[i], "\n")
    axs[i].errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)   #plot the data
    mu_fitted = np.zeros(1000)
    for j in range(len(coeff_ls[i])):
        mu_fitted += coeff_ls[i][j] * x**j
    axs[i].plot(x, mu_fitted, color='g')   #plot the polinomial fit on top
    axs[i].set_title("Fit with a " + str(i+1) + " degree polynomial")
    axs[i].set_xlabel("z")
    axs[i].set_ylabel("$\mu$")
    axs[i].set_xlim(0,2)
    axs[i].set_ylim(35,50);


#%%
### LINEAR REGRESSION ##########################################################################################

coeff_lr = []
model = LinearRegression()
model.fit(z_sample_sk, mu_sample, dmu)   #fitting a line
coeff_lr.append(model.coef_)

for i in range(1,max_deg):
    model = PolynomialRegression(i+1)
    model.fit(z_sample_sk, mu_sample, dmu)   #fitting polinomials with different degrees
    coeff_lr.append(model.coef_)

print("\n--- LINEAR REGRESSION ---")
print("    (blue plots)\n")
for i in range(max_deg):
    if print_pol_coeff:
        print(i+1, "degree polynomial:")
        print(coeff_lr[i], "\n")
    plt.figure()
    plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)   #plot the data
    mu_fitted = np.zeros(1000)
    for j in range(len(coeff_lr[i])):
        mu_fitted += coeff_lr[i][j] * x**j
    plt.plot(x, mu_fitted, color='dodgerblue')   #plot the polinomial fit on top
    plt.title("Fit with a " + str(i+1) + " degree polynomial")
    plt.xlabel("z")
    plt.ylabel("$\mu$")
    plt.xlim(0,2)
    plt.ylim(35,50)
plt.show()

#Since the results are pretty much identical, it's useless to fit them one on top of eachother


#%%
### BASIS FUNCTION REGRESSION ##################################################################################

print("\n--- BASIS FUNCTION REGRESSION ---")
print("    (red plots)\n")
x = np.linspace(0, 2, 1000)
for i in range(2, max_deg+2):
    plt.figure()
    plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)   #plot the data
    
    grid = np.linspace(0, 1, i)[:, np.newaxis]   #defining the mean of the gaussians
    sigma = 2*(grid[1] - grid[0])   #defining the sigma of the gaussians (twice the distance between two mu)
    model = BasisFunctionRegression('gaussian', mu=grid, sigma=sigma)
    model.fit(z_sample_sk, mu_sample, dmu)   #fitting the data
    mu_fitted = model.predict(x[:,np.newaxis])
    
    plt.plot(x, mu_fitted, color='crimson')   #plot the fit on top
    plt.title("Fit with " + str(i) + " gaussians")
    plt.xlabel("z")
    plt.ylabel("$\mu$")
    plt.xlim(0,2)
    plt.ylim(35,50)
plt.show()


#%%
### KERNEL REGRESSION ##########################################################################################

print("\n--- KERNEL REGRESSION ---")
print("    (yellow plots)\n")
x = np.linspace(0, 2, 1000)
for i in range(1, max_deg+1):
    plt.figure()
    plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)   #plot the data
    
    widht = 1/i   #defining the bandwidht
    model = NadarayaWatson(kernel='gaussian', h=widht)
    model.fit(z_sample_sk, mu_sample, dmu)   #fitting the data
    mu_fitted = model.predict(x[:,np.newaxis])
    
    plt.plot(x, mu_fitted, color='orange')   #plot the fit on top
    plt.title("Fit with kernel bandwidht " + str(np.round(1/i, 2)))
    plt.xlabel("z")
    plt.ylabel("$\mu$")
    plt.xlim(0,2)
    plt.ylim(35,50)
plt.show()


#%%
### CROSS VALIDATION ###########################################################################################

#dividing the data in training set and validation set
Xtrain, Xtest, ytrain, ytest, err_ytrain, err_ytest = train_test_split(z_sample, mu_sample, dmu, test_size=0.20)
Xtrain_sk = Xtrain[:, np.newaxis]
Xtest_sk = Xtest[:, np.newaxis]


### linear regression ###
TrainErr_lr = []
CvErr_lr = []
for i in range(max_deg):
    model = PolynomialRegression(i+1)
    model.fit(Xtrain_sk, ytrain, err_ytrain)   #fitting the train set
    TrainErr_lr.append(error(Xtrain_sk, ytrain, model))   #computing the training error
    CvErr_lr.append(error(Xtest_sk, ytest, model))   #computing the cross validation error
 
    
### basis function regression ###
TrainErr_bf = []
CvErr_bf = []
for i in range(2, max_deg+2):
    grid = np.linspace(0, 1, i)[:, np.newaxis]   #defining the mean of the gaussians
    sigma = 2*(grid[1] - grid[0])   #defining the sigma of the gaussians
    model = BasisFunctionRegression('gaussian', mu=grid, sigma=sigma)
    model.fit(Xtrain_sk, ytrain, err_ytrain)   #fitting the train set
    TrainErr_bf.append(error(Xtrain_sk, ytrain, model))   #computing the training error
    CvErr_bf.append(error(Xtest_sk, ytest, model))   #computing the cross validation error
    
    
### kernel regression ###
TrainErr_ker = []
CvErr_ker = []
for i in range(1, max_deg+1):
    widht = 1/i   #defining the bandwidht
    model = NadarayaWatson(kernel='gaussian', h=widht)
    model.fit(Xtrain_sk, ytrain, err_ytrain)   #fitting the train set
    TrainErr_ker.append(error(Xtrain_sk, ytrain, model))   #computing the training error
    CvErr_ker.append(error(Xtest_sk, ytest, model))   #computing the cross validation error


### plots ###
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
axs[0].plot(range(1, max_deg+1), TrainErr_lr, color='dodgerblue', label="Training err")
axs[0].plot(range(1, max_deg+1), CvErr_lr, color='dodgerblue', ls='dashed', label="CV error")
axs[1].plot(range(2, max_deg+2), TrainErr_bf, color='crimson', label="Training err")
axs[1].plot(range(2, max_deg+2), CvErr_bf, color='crimson', ls='dashed', label="CV error")
axs[2].plot(range(1, max_deg+1), TrainErr_ker, color='orange', label="Training err")
axs[2].plot(range(1, max_deg+1), CvErr_ker, color='orange', ls='dashed', label="CV error")
axs[0].set_ylim(0,2)
axs[0].set_title("Linear regression")
axs[1].set_title("Basis function regression")
axs[2].set_title("Kernel regression")
axs[0].set_ylabel("error")
axs[0].set_xlabel("degree of the polynomial")
axs[1].set_xlabel("number of gaussians")
axs[2].set_xlabel("inverse of the bandwidht of the kernel")
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()

#I notice it changes a lot depending on the separation between test and validation
#I do a Kfolding instead


#%%
### K FOLDING OF RGB ###########################################################################################

K = 20   #number of K-folds

### linear regression ###
TrainErr = []
CvErr = []
print("\n--- Kfold ---")
kf = KFold(n_splits=K, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(z_sample_sk)):
    if print_foldings:
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
    for l in tqdm(range(1,max_deg)):
        model = PolynomialRegression(l+1)
        model.fit(Xtrain, ytrain, err_ytrain)   #fitting the train set
        te.append(error(Xtrain, ytrain, model))   #computing the training error on this fold
        cve.append(error(Xtest, ytest, model))   #computing the cross validation error on this fold
    
    TrainErr.append(np.array(te))
    CvErr.append(np.array(cve))

TrainErr_med = np.median(TrainErr, axis=0)
CvErr_med = np.median(CvErr, axis=0)

plt.figure()
plt.plot(range(1,max_deg), TrainErr_med, color='dodgerblue', label="Training err")
plt.plot(range(1,max_deg), CvErr_med, color='dodgerblue', ls='dashed', label="CV error")
plt.xlabel('l')
plt.title("Kfold cross validation - polynomial regression")
plt.legend()
plt.show()


### basis function regression ###
TrainErr = []
CvErr = []
print("\n--- Kfold ---")
kf = KFold(n_splits=K, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(z_sample_sk)):
    if print_foldings:
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
    for l in tqdm(range(2, max_deg+2)):
        grid = np.linspace(0, 1, l)[:, np.newaxis]   #defining the mean of the gaussians
        sigma = 2*(grid[1] - grid[0])   #defining the sigma of the gaussians
        model = BasisFunctionRegression('gaussian', mu=grid, sigma=sigma)
        model.fit(Xtrain, ytrain, err_ytrain)   #fitting the train set
        te.append(error(Xtrain, ytrain, model))   #computing the training error on this fold
        cve.append(error(Xtest, ytest, model))   #computing the cross validation error on this fold
    
    TrainErr.append(np.array(te))
    CvErr.append(np.array(cve))

TrainErr_med = np.median(TrainErr, axis=0)
CvErr_med = np.median(CvErr, axis=0)

plt.figure()
plt.plot(range(2, max_deg+2), TrainErr_med, color='crimson', label="Training err")
plt.plot(range(2, max_deg+2), CvErr_med, color='crimson', ls='dashed', label="CV error")
plt.xlabel('l')
plt.title("Kfold cross validation - basis function regression")
plt.legend()
plt.show()


### kernel regression ###
TrainErr = []
CvErr = []
print("\n--- Kfold ---")
kf = KFold(n_splits=K, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(z_sample_sk)):
    if print_foldings:
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
    for l in tqdm(range(1, max_deg+1)):
        widht = 1/l   #defining the bandwidht
        model = NadarayaWatson(kernel='gaussian', h=widht)
        model.fit(Xtrain, ytrain, err_ytrain)   #fitting the train set
        te.append(error(Xtrain, ytrain, model))   #computing the training error on this fold
        cve.append(error(Xtest, ytest, model))   #computing the cross validation error on this fold
    
    TrainErr.append(np.array(te))
    CvErr.append(np.array(cve))

TrainErr_med = np.median(TrainErr, axis=0)
CvErr_med = np.median(CvErr, axis=0)

plt.figure()
plt.plot(range(1, max_deg+1), TrainErr_med, color='orange', label="Training err")
plt.plot(range(1, max_deg+1), CvErr_med, color='orange', ls='dashed', label="CV error")
plt.xlabel('l')
plt.xlim(max_deg+0.5,0.5)
plt.title("Kfold cross validation - kernel regression")
plt.legend()
plt.show()






