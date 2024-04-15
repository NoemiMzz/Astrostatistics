import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from astropy.visualization.hist import hist as fancyhist
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

################################################################################################################

choice_plots = True

################################################################################################################

N = 10000

sigma = 0.02
mu = 1   #normalized M_irr/M

### FUNCTIONS ##################################################################################################

def f(chi):
    return np.sqrt((1 + np.sqrt(1-chi**2)) / 2)

def pdf_f(f):
    return 2*(2*f**2-1)/(1 - f**2)**0.5

def kde_sklearn(xgrid, data, bandwidth = 1.0, kernel="linear"):
    kde_skl = KernelDensity(bandwidth = bandwidth, kernel=kernel)
    kde_skl.fit(data[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xgrid[:, np.newaxis]) # sklearn returns log(density)
    return np.exp(log_pdf)

#%%
### GENERATING DATA ############################################################################################

chi = np.random.uniform(0, 1, N)
x_f = np.sort(chi)

M_gauss = norm(loc=mu, scale=sigma)
M = M_gauss.rvs(N)


Mirr = f(chi) * M

x = np.linspace(Mirr.min(), Mirr.max(), 1000)

#%%
################################################################################################################

### f function ###
plt.figure()
plt.plot(x_f, f(x_f))
plt.title("f function")
plt.ylabel("f")
plt.xlabel("$\chi$")
plt.show()

plt.figure()
plt.hist(f(chi), bins=50, density=True, color='limegreen', alpha=0.3)
plt.plot(f(x_f), pdf_f(f(x_f)), color='limegreen')
plt.ylim(0, 40)
plt.title("f distribution")
plt.xlabel("f")
plt.show()

### mass distribution ###
plt.figure()
plt.hist(M, bins=50, density=True, color='deepskyblue', alpha=0.3)
plt.plot(x_f, M_gauss.pdf(x_f), color='deepskyblue')
plt.title("M distribution")
plt.xlabel("M")
plt.show()

### irriducible mass ###
plt.figure()
plt.hist(Mirr, bins=50, density=True, color='crimson', alpha=0.3)
#plt.plot(x_f, M_gauss.pdf(x_f), color='crimson')
plt.title("$M_{irr}$ distribution")
plt.xlabel("$M_{irr}$")
plt.show()

#%%
### KDE ########################################################################################################

### ch
bw = 0.03

PDFt = kde_sklearn(x, Mirr, bandwidth=bw, kernel="tophat")
PDFg = kde_sklearn(x, Mirr, bandwidth=bw, kernel="gaussian")
PDFe = kde_sklearn(x, Mirr, bandwidth=bw, kernel="epanechnikov")

if choice_plots :
    plt.figure()
    plt.hist(Mirr, histtype='step', density=True, color='black', linestyle='dotted')
    fancyhist(Mirr, bins='scott', histtype='step', density=True, color='black', linestyle='dashed')
    fancyhist(Mirr, bins='freedman', histtype='step', density=True, color='black')

    plt.plot(x, PDFt, color='red') 
    plt.plot(x, PDFg, color='red', linestyle='dashed')
    plt.plot(x, PDFe, color='red', linestyle='dotted')
    
    plt.title("Comparing different KDE")
    plt.xlabel("$M_{irr}$")
    plt.show()
    
plt.figure()
counts, bins, bars = fancyhist(Mirr, bins='scott', density=True, color='gray', alpha=0.3)
plt.plot(x, kde_sklearn(x, Mirr, bandwidth=0.001, kernel="epanechnikov"), color='gold', label="bw=0.001")
plt.plot(x, kde_sklearn(x, Mirr, bandwidth=0.01, kernel="epanechnikov"), color='red', label="bw=0.01")
plt.plot(x, kde_sklearn(x, Mirr, bandwidth=0.1, kernel="epanechnikov"), color='limegreen', label="bw=0.1")
plt.title("$M_{irr}$ distribution")
plt.xlabel("$M_{irr}$")
plt.legend()
plt.show()

cdf_norm = np.cumsum(counts) / np.sum(counts)
xgrid = (bins[1:] + bins[:-1]) / 2
plt.figure()
plt.plot(xgrid, cdf_norm, color="limegreen")
plt.title("$M_{irr}$ CDF")
plt.xlabel("$M_{irr}$")
plt.show()