import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from astropy.visualization.hist import hist as fancyhist
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp
from tqdm import tqdm

################################################################################################################

N = 10000

sigma = 0.02
mu = 1   #normalized M_irr/M

### FUNCTIONS ##################################################################################################

def f(chi):
    return np.sqrt((1 + np.sqrt(1-chi**2)) / 2)

def pdf_f(f):
    return 2*(2*f**2-1)/np.sqrt(1 - f**2)

def integrand(f, x):
    return ((2/np.pi)**0.5 / sigma ) * np.exp(-(x/f -1)**2 /(2*sigma**2)) * (2*f**2-1)/(1 - f**2)**0.5 / f

def pdf_Mirr(x):
    print("Integral:")
    return [quad(lambda f: integrand(f,xt), 1/2**0.5, 1)[0] for xt in tqdm(x)]

def kde_sklearn(xgrid, data, bandwidth, kernel):
    kde_skl = KernelDensity(bandwidth=bandwidth, kernel=kernel)   #define the KDE
    kde_skl.fit(data[:, np.newaxis])   #fit the data
    log_pdf = kde_skl.score_samples(xgrid[:, np.newaxis])   #compute the loglikelihood of each sample
    #the exponent of that it's the likelihood
    return np.exp(log_pdf)

#%%
### GENERATING DATA ############################################################################################

chi = np.random.uniform(0, 1, N)   #uniformely distributed chi

M_gauss = norm(loc=mu, scale=sigma)   #gaussian distributed masses (normalized)
M = M_gauss.rvs(N)

Mirr = f(chi) * M   #sample of irriducible masses

x = np.linspace(Mirr.min(), Mirr.max(), 1000)

#%%
################################################################################################################

### f function ###
plt.figure()
plt.plot(x, f(x), color='g')
plt.title("$f~(\\chi)$")
plt.ylabel("f")
plt.xlabel("$\chi$")
plt.show()

plt.figure()
f_axis = np.linspace(0.7, 1, 200)
plt.hist(f(chi), bins=50, density=True, color='limegreen', alpha=0.3, label="distribution")   #f(chi) distrib
plt.plot(f_axis, pdf_f(f_axis), color='limegreen', label="analytical pdf")   #analytical f(chi)
plt.ylim(0, 40)
plt.title("Distribution of $f~(\\chi)$")
plt.xlabel("f")
plt.legend()
plt.show()

### mass distribution ###
plt.figure()
x_M = np.linspace(0.8, 1.2, 200)
plt.hist(M, bins=50, density=True, color='deepskyblue', alpha=0.3, label="distribution")   #M distribution
plt.plot(x_M, M_gauss.pdf(x_M), color='deepskyblue', label="analytical pdf")   #analytical M
plt.xlim(0.8, 1.2)
plt.title("Distribution of masses")
plt.xlabel("M")
plt.legend()
plt.show()

### irriducible mass ###
plt.figure()
plt.hist(Mirr, bins=50, density=True, color='crimson', alpha=0.3, label="distribution")   #Mirr distribution
pdf_Mirr_analytical = pdf_Mirr(x)
plt.plot(x, pdf_Mirr_analytical, color='crimson', label="analytical pdf")
plt.title("$M_{irr}$ distribution")
plt.xlabel("$M_{irr}$")
plt.legend()
plt.show()


#%%
### RULES OF THUMB HIST ########################################################################################

### Mirr with different binning ###
plt.figure()
plt.hist(Mirr, bins=50, histtype='step', density=True, color='black')
fancyhist(Mirr, bins='scott', histtype='step', density=True, color='crimson', label="Scott")
fancyhist(Mirr, bins='freedman', histtype='step', density=True, color='orange', label="Freedman-Diaconis")
plt.xlabel("$M_{irr}$")
plt.title("Binning with different rules of thumb")
plt.legend()
plt.show()


#%%
### KDE ########################################################################################################

### different kernel density estimations ###
bw = 0.01

#compute the pdfs with different kernel functions
PDFt = kde_sklearn(x, Mirr, bandwidth=bw, kernel="tophat")
PDFg = kde_sklearn(x, Mirr, bandwidth=bw, kernel="gaussian")
PDFe = kde_sklearn(x, Mirr, bandwidth=bw, kernel="epanechnikov")

plt.figure()
plt.plot(x, pdf_Mirr_analytical, color='black', ls='dashed', label="analytical pdf")
plt.plot(x, PDFt, color='crimson', label="tophat kernel") 
plt.plot(x, PDFg, color='deepskyblue', label="gaussian kernel")
plt.plot(x, PDFe, color='orange', label="epanechnikov kernel")
plt.title("Comparing different KDE")
plt.xlabel("$M_{irr}$")
plt.legend()
plt.show()
    
#same kernel (gaussian), but changing the bandwidth
plt.figure()
plt.plot(x, pdf_Mirr_analytical, color='black', ls='dashed', label="analytical pdf")
plt.plot(x, kde_sklearn(x, Mirr, bandwidth=0.001, kernel="gaussian"), color='b', label="bw=0.001")
plt.plot(x, kde_sklearn(x, Mirr, bandwidth=0.01, kernel="gaussian"), color='deepskyblue', label="bw=0.01")
plt.plot(x, kde_sklearn(x, Mirr, bandwidth=0.1, kernel="gaussian"), color='c', label="bw=0.1")
plt.title("KDE with gaussian kernel")
plt.xlabel("$M_{irr}$")
plt.legend()
plt.show()


#%%
### SIGMA ANALYSIS #############################################################################################

sigma_range = np.logspace(-4, 4, 50)

### compute KS distances for different sigmas ###
KS_f = []
KS_M = []

for s in sigma_range:
    gauss_temp = norm(loc=mu, scale=s)
    M_var = gauss_temp.rvs(N)   #computing M distributions with different sigmas
    #computing the KS distances
    KS_f.append(ks_2samp(M_var * f(chi), f(chi)))
    KS_M.append(ks_2samp(M_var * f(chi), M_var))
    
KS_f = np.array(KS_f)
KS_M = np.array(KS_M)

#plot the results
plt.figure()
plt.plot(sigma_range, KS_f[:,0], color='limegreen', label='$M_{irr}$ vs f')
plt.plot(sigma_range, KS_M[:,0], color='deepskyblue', label='$M_{irr}$ vs M')
plt.xscale('log')
plt.xlabel("$\\sigma$")
plt.ylabel("KS")
plt.title("Kolmogorov-Smirnov distances")
plt.legend()
plt.show()


### plot limits ###
smallsigma = norm(loc=mu, scale=sigma_range[0]).rvs(N) * f(chi)
bigsigma = norm(loc=mu, scale=sigma_range[-1]).rvs(N) * f(chi)   #compute Mirr limit distributions

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
#sigma << mu
ax1.hist(smallsigma, bins=50, density=True, color='crimson', alpha=0.3, label="$M_{irr}$")
ax1.plot(f_axis, pdf_f(f_axis), color='limegreen', label="$f~(\\chi)$")   #analytical f(chi)
ax1.set_title("$\\sigma \\ll \\mu$")
ax1.legend()
#sigma >> mu
x2grid = np.linspace(-40000, 40000, 100)
ax2.hist(bigsigma, bins=50, density=True, color='crimson', alpha=0.3, label="$M_{irr}$")
plt.plot(x2grid, norm(loc=mu, scale=sigma_range[-1]).pdf(x2grid), color='deepskyblue', label="M")   #analytical M
ax2.set_title("$\\sigma \\gg \\mu$")
fig.suptitle("$\\sigma$ limits")
ax2.legend();

