import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

################################################################################################################

#always choose one :)
#np.random.seed(3)   #3 gaussians
np.random.seed(42)   #4 gaussians

################################################################################################################

### collect the data ###
data = np.load("../solutions/formationchannels.npy")   #(1,N) sklearn format
x = data.ravel()   #(N,1)

#raw plot
plt.figure()
plt.hist(x, bins=int(np.sqrt(len(x))), density=True, histtype='step', lw=2, color='royalblue')
plt.title("BH masses distribution")
plt.xlabel("mass")
plt.show()


### gaussian fit ###
xgrid = np.linspace(-10, 55, 100)   #x axes

num = np.arange(2, 11)   #number of gaussians fitted

aic = []

for n in tqdm(num):
    gm = GaussianMixture(n_components=n, covariance_type='diag').fit(data)   #fitting n gaussians
    means = gm.means_
    w = gm.weights_
    var = (gm.covariances_)**0.5
    
    aic.append(gm.aic(data))   #compute the Akaike Information Criterion
    
    plt.figure()
    plt.title("Fit with " + str(n) + " gaussians")
    plt.xlabel("mass")
    plt.hist(x, bins=int(np.sqrt(len(x))), density=True, color='gainsboro', alpha=0.7)   #plot the whole dataset
    
    for i in range(n) :
        distG = w[i] * norm(loc=means[i] , scale=var[i]).pdf(xgrid)   #plot each gaussian
        plt.plot(xgrid, distG)
        
    plt.show()


### model comparison with AIC ###    
best_index = np.argmin(aic)   #index of the #guassians that minimizes the AIC
best_num = best_index + 2   
    
plt.figure()
plt.title("Akaike Information Criterion")
plt.xlabel("#gaussians")
plt.ylabel("AIC")
plt.plot(num, aic, color='royalblue')
plt.scatter(best_index + 2, aic[best_index], marker='X', s=100, color='crimson')
plt.show()


#%%
### BEST FIT ###################################################################################################

### re-fit the gaussian misture with the best gaussians number ###
gm_best = GaussianMixture(n_components=best_num, covariance_type='diag').fit(data)
means_best = gm_best.means_
w_best = gm_best.weights_
var_best = (gm_best.covariances_)**0.5

pdf = np.exp( gm_best.score_samples(xgrid[:,np.newaxis]) )

plt.figure()
plt.hist(x, bins=int(np.sqrt(len(x))), density=True, color='gainsboro', alpha=0.7)   #plot of the whole dataset
for i in range(best_num):
    distG = w_best[i] * norm(loc=means_best[i] , scale=var_best[i]).pdf(xgrid)   #plot three gaussian
    plt.plot(xgrid, distG)
plt.plot(xgrid, pdf, color='black', ls='dashed')
plt.title("BH masses distribution fit")
plt.xlabel("mass")
plt.show

### compute which datapoints fit which guassian best ###
label = gm_best.predict(data)   #compute component density

#plot sorted data
c = ['orange', 'green', 'crimson', 'deepskyblue']
plt.figure()
for k in range(best_num) :
    plt.hist(x[label==k], density=True, color=c[k], alpha=0.5)
    distG = w_best[k] * norm(loc=means_best[k] , scale=var_best[k]).pdf(xgrid) * 5   #normalization by eye
    plt.plot(xgrid, distG, color=c[k])   #plot each gaussian
plt.title("BH masses distribution")
plt.xlabel("mass")
plt.show()

#The fit with 3 gaussians seems (by eye) slightly more accurate
#but the code better classifies which datapoint belongs to which gaussian with 4
#(still, not a wonderful classification)

