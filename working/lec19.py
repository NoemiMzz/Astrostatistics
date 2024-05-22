import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from astroML.classification import GMMBayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from tqdm import tqdm

### DATA #######################################################################################################

data = pd.read_csv("../solutions/galaxyquasar.csv")

ug = np.array(data.loc[:, 'u'] - data.loc[:, 'g']).squeeze()
gr = np.array(data.loc[:, 'g'] - data.loc[:, 'r']).squeeze()
ri = np.array(data.loc[:, 'r'] - data.loc[:, 'i']).squeeze()
iz = np.array(data.loc[:, 'i'] - data.loc[:, 'z']).squeeze()

lab = LabelEncoder()
# Assign 0 for galaxy and 1 for quasar
data['class'] = lab.fit_transform(data['class'])
labels = lab.inverse_transform(data['class'])

source = np.array(data.loc[:, 'class']).squeeze()
indQ = np.array(np.where(data.loc[:, 'class']==1)).squeeze()
indG = np.array(np.where(data.loc[:, 'class']==0)).squeeze()

N = len(source)

zoom = True
min_zoom = 0.85


#%%
### DATA PLOTS #################################################################################################

colors = np.array((ug, gr, ri, iz))
titles = ["U-G", "G-R", "R-I", "I-Z"]

### histograms ###
fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("Raw dataset", fontsize='14', y=0.96)

for i in range(4):
    axs[i].hist(colors[i][indQ], bins=int(np.sqrt(N)), density=True, histtype='step', lw=2, color='c', label="quasars")
    axs[i].hist(colors[i][indG], bins=int(np.sqrt(N)), density=True, histtype='step', lw=2, color='orchid', label="galaxies")
    axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[i].set_title(titles[i])
  
axs[0].set_xlim(-0.5, 2.5)
axs[3].legend()


#%%
### NAIVE BAYES ################################################################################################

### ROC curve ###
Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("ROC - Naive Bayes", fontsize='14', y=0.96)

clf = GaussianNB()
print("\n--- Naive Bayes ---")
for i in tqdm(range(4)):
    clf.fit(Xtrain[:,i][:, np.newaxis], ytrain)
    
    yprob = clf.predict_proba(Xtest[:,i][:, np.newaxis])
    
    fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])
    
    axs[i].plot(fpr, tpr, lw=2, color='navy')
    axs[i].set_title(titles[i])

if zoom:
    axs[0].set_ylim(min_zoom, 1)    
axs[2].set_xlabel("false positive rate") 
axs[3].set_xlabel("false positive rate")    
axs[0].set_ylabel("true positive rate") 
axs[2].set_ylabel("true positive rate")     
    
#the only color for which we have a good ROC curve is the U-G one


### plot results ###
clf.fit(ug[:, np.newaxis], source)
ypred = clf.predict(ug[:, np.newaxis])
indG_pred = np.array(np.where(ypred==0)).squeeze()
indQ_pred = np.array(np.where(ypred==1)).squeeze()

plt.figure()
#raw data
plt.hist(ug[indQ], bins=int(np.sqrt(N)), density=True, histtype='step', lw=2, color='c', label="quasars")
plt.hist(ug[indG], bins=int(np.sqrt(N)), density=True, histtype='step', lw=2, color='orchid', label="galaxies")
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#from classification
plt.hist(ug[indQ_pred], bins=int(np.sqrt(N)), density=True, alpha=0.5, color='c', label="quasars prediction")
plt.hist(ug[indG_pred], bins=int(np.sqrt(N)), density=True, alpha=0.5, color='orchid', label="galaxies prediction")
plt.xlim(-0.5, 2.5)
plt.title("Naive Bayes with U-G")
plt.legend()
plt.show()





#%%
### QUADRATIC DISCRIMINANT ANALYSIS ############################################################################

### ROC curve ###
Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("ROC - Quadratic Discriminant Analysis", fontsize='14', y=0.96)

qda = QDA()
print("\n--- Quadratic Discriminant Analysis ---")
for i in tqdm(range(4)):
    qda.fit(Xtrain[:,i][:, np.newaxis], ytrain)
    
    yprob = qda.predict_proba(Xtest[:,i][:, np.newaxis])
    
    fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])
    
    axs[i].plot(fpr, tpr, lw=2, color='navy')
    axs[i].set_title(titles[i])
    
if zoom:
    axs[0].set_ylim(min_zoom, 1)       
axs[2].set_xlabel("false positive rate") 
axs[3].set_xlabel("false positive rate")    
axs[0].set_ylabel("true positive rate") 
axs[2].set_ylabel("true positive rate")    


#%%
### GMM BAYES CLASSIFIER #######################################################################################

### ROC curve ###
Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("ROC - GMM Bayes Classifier", fontsize='14', y=0.96)

gmmb = GMMBayes(n_components=4)
print("\n--- GMM Bayes Classifier ---")
for i in tqdm(range(4)):
    gmmb.fit(Xtrain[:,i][:, np.newaxis], ytrain)
    
    yprob = gmmb.predict_proba(Xtest[:,i][:, np.newaxis])
    
    fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])
    
    axs[i].plot(fpr, tpr, lw=2, color='navy')
    axs[i].set_title(titles[i])
    
if zoom:
    axs[0].set_ylim(min_zoom, 1)    
axs[2].set_xlabel("false positive rate") 
axs[3].set_xlabel("false positive rate")    
axs[0].set_ylabel("true positive rate") 
axs[2].set_ylabel("true positive rate")   


#%%
### K-NEAREST NEIGHBOR CLASSIFIER ##############################################################################

### ROC curve ###
Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("ROC - K-Nearest Neighbor Classifier", fontsize='14', y=0.96)

knc = KNeighborsClassifier(n_neighbors=10)
print("\n--- K-Nearest Neighbor Classifier ---")
for i in tqdm(range(4)):
    knc.fit(Xtrain[:,i][:, np.newaxis], ytrain)
    
    yprob = knc.predict_proba(Xtest[:,i][:, np.newaxis])
    
    fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])
    
    axs[i].plot(fpr, tpr, lw=2, color='navy')
    axs[i].set_title(titles[i])

if zoom:
    axs[0].set_ylim(min_zoom, 1)    
axs[2].set_xlabel("false positive rate") 
axs[3].set_xlabel("false positive rate")    
axs[0].set_ylabel("true positive rate") 
axs[2].set_ylabel("true positive rate")     

