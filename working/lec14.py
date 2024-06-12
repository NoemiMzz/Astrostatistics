import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import warnings

################################################################################################################

conceal_warnings = True

if conceal_warnings:
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

### DATA AND PARAMETERS ########################################################################################

# Download file
r = requests.get('https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt')
with open("Summary_table.txt", 'wb') as f:
    f.write(r.content)

# Read content
data = np.loadtxt("Summary_table.txt", dtype='str',unpack='True')

# Read headers
with open("Summary_table.txt",'r') as f:
    names = np.array([n.strip().replace(" ","_") for n in f.readlines()[1].replace("#","").replace("\n","").lstrip().split('    ') if n.strip()!=''])

# Variable stored in the i-th index
print_names = True

if print_names:
    print("In the dataset we find:")
    for i in range(np.shape(data)[0]):
        print(i, "\t", names[i])

N = np.shape(data)[1]   #number of bursts

################################################################################################################

T90 = np.array(data[6], dtype=float)
T100 = np.array(data[12], dtype=float)
fluence = np.array(data[9], dtype=float)

#plot the two time distributions
#they're more or less the same
plt.figure()
plt.hist(T90, bins=np.logspace(-2,3,50), histtype='step', lw=2, label="T90", color='navy')
plt.hist(T100, bins=np.logspace(-2,3,50), histtype='step', lw=2, label="T100", color='royalblue')
plt.semilogx()
plt.xlabel("time (s)")
plt.title("Time distribution")
plt.legend()
plt.show()

#I choose to explore the T90 times vs the flux
plt.figure()
plt.scatter(T90, fluence, s=5, alpha=0.5, label="T90", color='navy')
plt.yscale('log')
plt.xscale('log')
plt.xlabel("time (s)")
plt.ylabel("fluence (erg/$cm^2$)")
plt.title("90% of photon counts")
plt.legend()
plt.show()


### data in sklearn format ###
T90sk = T90[:, np.newaxis]
fluencesk = fluence[:, np.newaxis]
Xtot = np.concatenate((np.log10(T90sk), np.log10(fluencesk)), axis=1)

#data cleaning
X = []
for i in range(len(Xtot)):
    if ( np.isfinite(Xtot[i,0]) and np.isfinite(Xtot[i,1]) ):   #I want to remove the nan and inf points
        X.append(Xtot[i])
X = np.array(X, dtype='float64')


#%%
### KMEANS #####################################################################################################

clf = KMeans(n_clusters=2)   # 2 clusters
clf.fit(X)
centersKM = clf.cluster_centers_   #location of the clusters
labelsKM = clf.predict(X)   #labels for each of the points

#labeling correctly the clusters
if centersKM[0,0] > centersKM[1,0]:
    namesKM = ["long gamma-ray bursts", "short gamma-ray bursts"]
    colorsKM = ['crimson', 'orange']
else:
    namesKM = ["short gamma-ray bursts", "long gamma-ray bursts"]
    colorsKM = ['orange', 'crimson']

#plot the data color-coded by cluster id
plt.figure()
for i in range(2):
    plt.scatter(X[labelsKM==i,0], X[labelsKM==i,1], s=5, alpha=0.3, label=namesKM[i], color=colorsKM[i])
plt.scatter(centersKM[:, 0], centersKM[:, 1], marker='x', color='black')
plt.xlabel("time (s)")
plt.ylabel("fluence (erg/$cm^2$)")
plt.title("90% of photon counts")
plt.legend()
plt.show()

#visualize the clustering in the time space
plt.figure()
for i in range(2):
    plt.hist(X[labelsKM==i,0], bins=np.linspace(-2,3,50), alpha=0.5, label=namesKM[i], color=colorsKM[i])
    plt.axvline(centersKM[i,0], color=colorsKM[i], linestyle='dashed')
plt.hist(np.log10(T90), bins=np.linspace(-2,3,50), histtype='step', lw=2, label="total", color='navy')
plt.xlim(-2, 3)
plt.xlabel("time (s)")
plt.title("Time distribution")
plt.legend()
plt.show()


#%%
### MEANSHIFT ##################################################################################################

scaler = preprocessing.StandardScaler()   #if I don't scale the data MS doesn't work properly
X_scaled = scaler.fit_transform(X)

bandwidth = 0.3
ms = MeanShift(bandwidth=bandwidth, seeds=scaler.fit_transform(centersKM))
ms.fit(X_scaled)
centersMS = scaler.inverse_transform(ms.cluster_centers_)   #location of the clusters
labelsMS = ms.labels_   #labels for each of the points
C = len(np.unique(labelsMS))

if C==2:

    #labeling correctly the clusters
    if centersMS[0,0] > centersMS[1,0]:
        namesMS = ["long gamma-ray bursts", "short gamma-ray bursts"]
        colorsMS = ['crimson', 'orange']
    else:
        namesMS = ["short gamma-ray bursts", "long gamma-ray bursts"]
        colorsMS = ['orange', 'crimson']
    
    #plot the data color-coded by cluster id
    plt.figure()
    for i in range(C):
        plt.scatter(X[labelsMS==i,0], X[labelsMS==i,1], s=5, alpha=0.3, label=namesMS[i], color=colorsMS[i])
    plt.scatter(centersMS[:, 0], centersMS[:, 1], marker='x', color='black')
    plt.xlabel("time (s)")
    plt.ylabel("fluence (erg/$cm^2$)")
    plt.title("90% of photon counts")
    plt.legend()
    plt.show()
    
    #visualize the clustering in the time space
    plt.figure()
    for i in range(C):
        plt.hist(X[labelsMS==i,0], bins=np.linspace(-2,3,50), alpha=0.5, label=namesMS[i], color=colorsMS[i])
        plt.axvline(centersMS[i,0], color=colorsMS[i], linestyle='dashed')
    plt.hist(np.log10(T90), bins=np.linspace(-2,3,50), histtype='step', lw=2, label="total", color='navy')
    plt.xlim(-2, 3)
    plt.xlabel("time (s)")
    plt.title("Time distribution")
    plt.legend()
    plt.show()

#code adaptation in case MS finds more than two clusters
#(usually when I don't suggest the centroids from the previous analysis)    
else:
    print("---------------------------------------------------------")
    if C>2:
        print("More than two clusters")
    elif C<2:
        print("Less than two clusters")
    print("Data cannot be clustered as long or short gamma-ray burst")
    print("---------------------------------------------------------\n")
    
    #plot the data
    plt.figure()
    for i in range(C):
        plt.scatter(X[labelsMS==i,0], X[labelsMS==i,1], s=5, alpha=0.5)
    plt.scatter(centersMS[:, 0], centersMS[:, 1], marker='x', color='black')
    plt.xlabel("time (s)")
    plt.ylabel("fluence (erg/$cm^2$)")
    plt.title("90% of photon counts")
    plt.show()
    
    plt.figure()
    for i in range(C):
        plt.hist(X[labelsMS==i,0], bins=np.linspace(-2,3,50), alpha=0.5)
    plt.hist(np.log10(T90), bins=np.linspace(-2,3,50), histtype='step', lw=2, label="total", color='navy')
    plt.xlim(-2, 3)
    plt.xlabel("time (s)")
    plt.title("Time distribution")
    plt.show()


#%%
### COMMENTS ###################################################################################################

### observation frequency ###
N0 = np.bincount(labelsKM)[0]
N1 =  np.bincount(labelsKM)[1] 
print("\nWith KMeans we find:")
print(np.round(N0/N*100, 1), "% of", namesKM[0])
print(np.round(N1/N*100, 1), "% of", namesKM[1])
# long gamma-ray bursts are more likely to be observed
# maybe an even better clustering can be obtained by equally weighting the two populations


### methods comparison ###
# KMeans works beautyfully
# Meanshift needs a little more adjustments (rescaling the data, suggesting the centroids)
# the centroids seem to be computed better in KMeans

