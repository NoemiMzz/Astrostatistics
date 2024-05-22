import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn import preprocessing

### DATA AND PARAMETERS ########################################################################################

# Download file
r = requests.get('https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt')
with open("Summary_table.txt", 'wb') as f:
    f.write(r.content)

# Read content
data = np.loadtxt("Summary_table.txt", dtype='str',unpack='True')

# Read headers
with open("Summary_table.txt",'r') as f:
    names= np.array([n.strip().replace(" ","_") for n in f.readlines()[1].replace("#","").replace("\n","").lstrip().split('    ') if n.strip()!=''])

# Variable stored in the i-th index
print_names = True
if print_names:
    for i in range(np.shape(data)[0]):
        print(i, "\t", names[i])

N = np.shape(data)[1]   #number of bursts

################################################################################################################

T90 = np.array(data[6], dtype=float)
T100 = np.array(data[12], dtype=float)
fluence = np.array(data[9], dtype=float)

plt.figure()
plt.hist(T90, bins=np.logspace(-2,3,50), histtype='step', lw=2, label="T90", color='navy')
plt.hist(T100, bins=np.logspace(-2,3,50), histtype='step', lw=2, label="T100", color='royalblue')
plt.semilogx()
plt.xlabel("time (s)")
plt.title("Time distribution")
plt.legend()
plt.show()

plt.figure()
plt.scatter(T90, fluence, s=5, alpha=0.5, label="T90", color='navy')
plt.yscale('log')
plt.xscale('log')
plt.xlabel("time (s)")
plt.ylabel("fluence (erg/$cm^2$)")
plt.title("90% of photon counts")
plt.legend()
plt.show()

plt.figure()
plt.scatter(T100, fluence, s=5, alpha=0.5, label="T100", color='royalblue')
plt.yscale('log')
plt.xscale('log')
plt.xlabel("time (s)")
plt.ylabel("fluence (erg/$cm^2$)")
plt.title("100% of photon counts")
plt.legend()
plt.show()

#%%
### data in sklearn format ###
T90sk = T90[:, np.newaxis]
fluencesk = fluence[:, np.newaxis]
Xtot = np.concatenate((np.log10(T90sk), np.log10(fluencesk)), axis=1)

X = []
for i in range(len(Xtot)):
    if ( np.isfinite(Xtot[i,0]) and np.isfinite(Xtot[i,1]) ):   #I want to remove the nan and inf points
        X.append(Xtot[i])
X = np.array(X, dtype='float64')


#%%
### clustering with KMeans ###
clf = KMeans(n_clusters=2)   # 2 clusters
clf.fit(X)
centersKM = clf.cluster_centers_   #location of the clusters
labelsKM = clf.predict(X)   #labels for each of the points

#labeling correctly the clusters
if centersKM[0,0] > centersKM[1,0]:
    names = ["long gamma-ray bursts", "short gamma-ray bursts"]
    colors = ['crimson', 'orange']
else:
    names = ["short gamma-ray bursts", "long gamma-ray bursts"]
    colors = ['orange', 'crimson']

#plot the data color-coded by cluster id
plt.figure()
for i in range(2):
    plt.scatter(X[labelsKM==i,0], X[labelsKM==i,1], s=5, alpha=0.5, label=names[i], color=colors[i])
plt.scatter(centersKM[:, 0], centersKM[:, 1], marker='x', color='black')
plt.xlabel("time (s)")
plt.ylabel("fluence (erg/$cm^2$)")
plt.title("90% of photon counts")
plt.legend()
plt.show()

plt.figure()
for i in range(2):
    plt.hist(X[labelsKM==i,0], bins=np.linspace(-2,3,50), alpha=0.5, label=names[i], color=colors[i])
plt.hist(np.log10(T90), bins=np.linspace(-2,3,50), histtype='step', lw=2, label="total", color='navy')
plt.xlim(-2, 3)
plt.xlabel("time (s)")
plt.title("Time distribution")
plt.legend()
plt.show()


#%%
### clustering with MeanShift ###
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
        names = ["long gamma-ray bursts", "short gamma-ray bursts"]
        colors = ['crimson', 'orange']
    else:
        names = ["short gamma-ray bursts", "long gamma-ray bursts"]
        colors = ['orange', 'crimson']
    
    #plot the data color-coded by cluster id
    plt.figure()
    for i in range(C):
        plt.scatter(X[labelsMS==i,0], X[labelsMS==i,1], s=5, alpha=0.5, label=names[i], color=colors[i])
    plt.scatter(centersMS[:, 0], centersMS[:, 1], marker='x', color='black')
    plt.xlabel("time (s)")
    plt.ylabel("fluence (erg/$cm^2$)")
    plt.title("90% of photon counts")
    plt.legend()
    plt.show()
    
    plt.figure()
    for i in range(C):
        plt.hist(X[labelsMS==i,0], bins=np.linspace(-2,3,50), alpha=0.5, label=names[i], color=colors[i])
    plt.hist(np.log10(T90), bins=np.linspace(-2,3,50), histtype='step', lw=2, label="total", color='navy')
    plt.xlim(-2, 3)
    plt.xlabel("time (s)")
    plt.title("Time distribution")
    plt.legend()
    plt.show()
    
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

