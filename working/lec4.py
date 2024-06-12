import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ecdf
from scipy.stats import ks_2samp
from astroML.datasets import fetch_dr7_quasar
from tqdm import tqdm

################################################################################################################

N = 100000
b = 24   #number of bins

#%%
### DATA FROM SDSS #############################################################################################

# Fetch the quasar data
data = fetch_dr7_quasar()

# select the first 10000 points
data = data[:10000]

z = data['redshift']

# quasars histogram
plt.figure()
hights, bins, patches = plt.hist(z, bins=b, density=True, color="deepskyblue", label="SSDS data")

#%%
### REJECTION ##################################################################################################

print("\n- REJECTION SAMPLING -")

#generating uniform points
u_x = np.random.uniform(bins[0], bins[b], N)
u_y = np.random.uniform(0, max(hights), N)


### point selection ###
clone = []
for n in tqdm(range(N)) :
    for i in range(b) :
        if bins[i] < u_x[n] < bins[i+1] :   #if the x is in the bin
            if u_y[n] < hights[i] :   #and the y is under the hist
                clone.append(u_x[n])   #select the point

#plot and confront with the given data
plt.hist(clone, bins=b, density=True, histtype="step", lw=2, color="royalblue", label="Cloned data")
plt.legend()
plt.title("Rejection sampling")
plt.xlabel("redshift")
plt.show()


### distributions comparison ###
print(ks_2samp(z, clone))

#%%
### INVERSE TRANSFORM ##########################################################################################

print("\n- INVERSE TRANSFORM SAMPLING -")

### cdf computation and inversion ###
res = ecdf(z)   #empirical cdf
z_cdf = res.cdf.quantiles
prob_cdf = res.cdf.probabilities

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3.5))
fig.suptitle("Cumulative density function")
ax1.plot(z_cdf, prob_cdf, color="green", lw=2, label="cdf")   #cdf
ax1.set_xlabel("redshift")
ax1.legend()
ax2.plot(prob_cdf, z_cdf, color="orange", lw=2, label="cdf$^{-1}$")   #inverse of cdf
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel("redshift")
ax2.legend()


### sample from cdf ###
sample_x = np.random.uniform(0, 1, N)   #random x-values of the quantile function picked uniformely
sample_z = np.interp(sample_x, prob_cdf, z_cdf)   #corresponding redshift values

# plot and confront with the given data
plt.figure()
plt.hist(z, bins=b, density=True, color="deepskyblue", label="SSDS data")
plt.hist(sample_z, bins=b, density=True, histtype="step", lw=2, color="royalblue", label="Cloned data")
plt.legend()
plt.title("Inverse transform sampling")
plt.xlabel("redshift")
plt.show()


### distributions comparison ###
print(ks_2samp(z, sample_z))
