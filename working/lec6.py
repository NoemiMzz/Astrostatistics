import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

################################################################################################################

### collect the data ###
data = np.load("../solutions/formationchannels.npy")   #(1,N)
x = data.ravel()   #(N,1)

plt.figure()
plt.hist(x, bins=int(np.sqrt(len(x))), density=True, color='royalblue')
plt.title("BH masses distribution")
plt.xlabel("mass")
plt.show()


### gaussian fit ###
xgrid = np.linspace(-10, 55, 100)   #x axes

num = np.arange(2, 11)   #number of gaussians fitted

aic = []

for n in tqdm(num):
    gm = GaussianMixture(n_components=n, covariance_type='diag').fit(data)
    means = gm.means_
    w = gm.weights_
    var = (gm.covariances_)**0.5
    
    aic.append(gm.aic(data))   #compute the Akaike Information Criterion
    
    plt.figure()
    plt.title("Fit with " + str(n) + " gaussians")
    plt.xlabel("mass")
    plt.hist(x, bins=int(np.sqrt(len(x))), density=True, color='lightgray')   #plot of the whole dataset
    
    for i in range(n) :
        distG = w[i] * norm(loc=means[i] , scale=var[i]).pdf(xgrid)   #plot each gaussian
        plt.plot(xgrid, distG)
        
    plt.show()


### model comparison with AIC ###    
best_num = np.argmin(aic)   #index of the #guassians that minimizes the AIC
    
plt.figure()
plt.title("Akaike Information Criterion")
plt.xlabel("#gaussians")
plt.ylabel("AIC")
plt.plot(num, aic, color='royalblue')
plt.scatter(best_num + 2, aic[best_num], marker='X', s=100, color='crimson')
plt.show()

#%%
################################################################################################################

if (best_num + 2) == 3 :
    ### re-fit the gaussian misture with 3 gaussians ###
    gm_best = GaussianMixture(n_components=3).fit(data)
    means_best = gm_best.means_
    w_best = gm_best.weights_
    var_best = (gm_best.covariances_)**0.5
    
    ### compute which datapoints fit which guassian best ###
    prob = gm_best.predict_proba(data)   #compute component density
    
    ga = []
    gb = []
    gc = []
    
    for j in tqdm(range(len(data))) :   #sort the data depending which gaussian is more probable
        t = np.argmax(prob[j])
        if t == 0 :
            ga.append(x[j])
        if t == 1 :
            gb.append(x[j])
        if t == 2 :
            gc.append(x[j])
    
    #plot sorted data
    c = ['orange', 'green', 'crimson']
    plt.figure()
    plt.hist(ga, density=True, color=c[0], alpha=0.5)
    plt.hist(gb, density=True, color=c[1], alpha=0.5)
    plt.hist(gc, density=True, color=c[2], alpha=0.5)
    for k in range(3) :
        distG_best = norm(loc=means_best[k] , scale=var_best[k]).pdf(xgrid)   #plot each gaussian
        plt.plot(xgrid, distG_best.squeeze(), color=c[k])
    plt.title("BH masses distribution")
    plt.xlabel("mass")
    plt.show()
    
else :
    print("Shouldn't the best fit be with 3 gaussians? Try running again")