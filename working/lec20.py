import numpy as np
import matplotlib.pyplot as plt
import h5py
import corner
from ipywidgets import interact
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


#%%
### DATA #######################################################################################################

data = h5py.File("../../sample_2e7_design_precessing_higherordermodes_3detectors.h5", "r")
#print(list(data.keys()))
N = len(data['q'])


#%%
### FUNCTIONS ##################################################################################################

from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[0, 1000, 
                                            0, 4], 
                           alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 
             "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 
             "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    
    
#%%
### DOWNSAMPLING ###############################################################################################

### downsampling ###
step = 10000
ind = np.arange(0, N, step)   #selected indices
print("\nIn this analysis I will use", len(ind), "datapoints")

#organize the downsampled dataset
mtot = data['mtot'][ind]
q = data['q'][ind]
i = data['iota'][ind]
psi = data['psi'][ind]
z = data['z'][ind]
s1x = data['chi1x'][ind]
s1y = data['chi1y'][ind]
s1z = data['chi1z'][ind]
s2x = data['chi2x'][ind]
s2y = data['chi2y'][ind]
s2z = data['chi2z'][ind]
snr = data['det'][ind]
features = np.array([mtot, q, i, psi, z, s1x, s1y, s1z, s2x, s2y, s2z])
labels = ['mtot', 'q', 'i', 'psi', 'z', 'spin 1x', 'spin 1y', 'spin 1z', 'spin 2x', 'spin 2y', 'spin 2z']

#plotting all the features
fig = corner.corner(features.T, labels=labels, levels=[0.68,0.95], color='navy')
fig.suptitle("Raw data downsampled");

#selecting the indices with SNR > 12
s_good = (ind[np.where(snr==1)] / step).astype(int)
s_bad = (ind[np.where(snr==0)] / step).astype(int)


#%%
### PCA ########################################################################################################

print("\n\n--- PCA ---")

features_scaled = (StandardScaler().fit_transform(features)).T

### pca ###
pca = PCA(n_components=3)   #running the PCA
pca.fit(features_scaled)

proj = pca.transform(features_scaled)   #projection onto the new features
info = pca.explained_variance_ratio_   #information of each new feature
composition = pca.components_   #which old component is prevalent


fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()
axs[0].scatter(proj[:,0][s_bad], proj[:,1][s_bad], color='royalblue', label="SNR < 12", alpha=0.5)
axs[0].scatter(proj[:,0][s_good], proj[:,1][s_good], color='crimson', label="SNR > 12", alpha=0.5)
#axs[0].set_xscale('log')
#axs[0].set_yscale('log')
axs[0].set_xlabel("PCA1")
axs[0].set_ylabel("PCA2")

axs[1].scatter(proj[:,1][s_bad], proj[:,2][s_bad], color='royalblue', label="SNR < 12", alpha=0.5)
axs[1].scatter(proj[:,1][s_good], proj[:,2][s_good], color='crimson', label="SNR > 12", alpha=0.5)
#axs[1].set_xscale('log')
#axs[1].set_yscale('log')
axs[1].set_xlabel("PCA2")
axs[1].set_ylabel("PCA3")

axs[2].scatter(proj[:,2][s_bad], proj[:,0][s_bad], color='royalblue', label="SNR < 12", alpha=0.5)
axs[2].scatter(proj[:,2][s_good], proj[:,0][s_good], color='crimson', label="SNR > 12", alpha=0.5)
#axs[2].set_xscale('log')
#axs[2].set_yscale('log')
axs[2].set_xlabel("PCA3")
axs[2].set_ylabel("PCA1");


for i in range(3):
    print("\n- PCA"+str(i+1)+" contains", np.around(info[i]*100, 1), "% of the information")
    print("  and it's composed by: ")
    for j in range(len(labels)):
          print("  ", str(labels[j])+":", np.around(composition[i, j], 3))  
print("\nWith 3 components I obtain", np.around(np.sum(info)*100, 1), "% of the information")

ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
ax.scatter(proj[:,0][s_bad], proj[:,1][s_bad], proj[:,2][s_bad], label="SNR < 12",
           color='royalblue', s=5, alpha=0.5)
ax.scatter(proj[:,0], proj[:,1], proj[:,2], c=snr,
           cmap='coolwarm', s=5, alpha=0.5)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
#ax.set_title("")
plt.show()



#%%
### RANDOM FOREST ##############################################################################################

plt.figure()
plt.scatter(mtot[s_bad], z[s_bad], color='royalblue', label="SNR < 12")
plt.scatter(mtot[s_good], z[s_good], color='crimson', label="SNR > 12")
plt.xlabel("$m_{tot}$")
plt.ylabel("redshift")
plt.legend(loc='lower right')
plt.show()


X = np.array([mtot, z]).T
rf = RandomForestClassifier(10)
rf.fit(X, snr)
snr_pred = rf.predict(X)

s_good_pred = (ind[np.where(snr_pred==1)] / step).astype(int)
s_bad_pred = (ind[np.where(snr_pred==0)] / step).astype(int)

plt.figure()
plt.scatter(mtot[s_bad], z[s_bad], color='royalblue', s=50, alpha=0.3)
plt.scatter(mtot[s_good], z[s_good], color='crimson', s=50, alpha=0.3)
plt.scatter(mtot[s_bad_pred], z[s_bad_pred], marker='x', color='royalblue', s=15, label="SNR < 12")
plt.scatter(mtot[s_good_pred], z[s_good_pred], marker='x', color='crimson', s=15, label="SNR > 12")
plt.xlabel("$m_{tot}$")
plt.ylabel("redshift")
plt.legend(loc='lower right')
plt.show()
