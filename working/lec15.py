from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

### DATA AND PARAMETERS ########################################################################################

urllib.request.urlretrieve("https://raw.githubusercontent.com/nshaud/ml_for_astro/main/stars.csv", "stars.csv")
df_stars = pd.read_csv("stars.csv")

le = LabelEncoder()
# Assign unique integers from 0 to 6 to each star type
df_stars['Star type'] = le.fit_transform(df_stars['Star type'])
labels = le.inverse_transform(df_stars['Star type'])

#%%
### FUNCTIONS ##################################################################################################

def project(evals, evecs, mean):
    return mean + np.dot(evals, evecs)

#%%
################################################################################################################

### usual HR diagram ###
fig = plt.figure(figsize=(7, 7))
sns.scatterplot(data=df_stars, x='Temperature (K)', y='Luminosity(L/Lo)', hue=labels)
plt.xscale('log')
plt.yscale('log')
plt.xticks([5000, 10000, 50000])
plt.xlim(5e4, 1.5e3)
plt.show()


### new HR disgram ###
panda = df_stars.loc[:, 'Temperature (K)':'Absolute magnitude(Mv)']   #selecting data
panda = np.array(panda)
X = StandardScaler().fit_transform(panda)   #rescaling the data

plt.figure()
plt.scatter(X[:,0], X[:,3], c=df_stars.loc[:, 'Star type'], s=8, cmap='rainbow')
plt.show()

pca = PCA(n_components=2)   #running the PCA
pca.fit(X)

evals = pca.transform(X)
mean = pca.mean_
evecs = pca.components_

plt.figure()
plt.scatter(evals[:,0], evals[:,1], c=df_stars.loc[:, 'Star type'], s=8, cmap='rainbow')
plt.show()

proj = project(evals, evecs, mean)
plt.figure()
plt.scatter(proj[:,0], proj[:,3], c=df_stars.loc[:, 'Star type'], s=8, cmap='rainbow')
plt.show()
