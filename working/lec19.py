import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from astroML.classification import GMMBayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve

### DATA AND PARAMETERS ########################################################################################

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

#to zoom the y-axis in the ROC curve
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

#from the plot we can see that in U-G the two sources are well separated


#%%
### NAIVE BAYES ################################################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("ROC - Naive Bayes", fontsize='14', y=0.96)

clf = GaussianNB()   #choosen classifier
print("\n\n--- Naive Bayes ---")
for i in range(4):
    clf.fit(Xtrain[:,i][:, np.newaxis], ytrain)   #I train each color separately
    
    yprob = clf.predict_proba(Xtest[:,i][:, np.newaxis])
    ypred = clf.predict(Xtest[:,i][:, np.newaxis])
    
    fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #compute true and false positive rate
    
    axs[i].plot(fpr, tpr, lw=2, color='navy')   #plot the ROC curve
    axs[i].set_title(titles[i])
    
    print("\n", titles[i])
    print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
    print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
    print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly

if zoom:
    axs[0].set_ylim(min_zoom, 1)    
axs[2].set_xlabel("false positive rate") 
axs[3].set_xlabel("false positive rate")    
axs[0].set_ylabel("true positive rate") 
axs[2].set_ylabel("true positive rate");    


### plot results for U-G ###
Xtrain, Xtest, ytrain, ytest = train_test_split(ug[:, np.newaxis], source, test_size=0.30)
clf.fit(Xtrain, ytrain)

ug_pred = clf.predict(ug[:, np.newaxis])
indG_pred = np.array(np.where(ug_pred==0)).squeeze()
indQ_pred = np.array(np.where(ug_pred==1)).squeeze()

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

Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("ROC - Quadratic Discriminant Analysis", fontsize='14', y=0.96)

qda = QDA()   #choosen classifier
print("\n\n--- Quadratic Discriminant Analysis ---")
for i in range(4):
    qda.fit(Xtrain[:,i][:, np.newaxis], ytrain)   #I train each color separately
    
    yprob = qda.predict_proba(Xtest[:,i][:, np.newaxis])
    ypred = qda.predict(Xtest[:,i][:, np.newaxis])
    
    fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #compute true and false positive rate
    
    axs[i].plot(fpr, tpr, lw=2, color='navy')   #plot the ROC curve
    axs[i].set_title(titles[i])
    
    print("\n", titles[i])
    print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
    print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
    print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly
    
if zoom:
    axs[0].set_ylim(min_zoom, 1)       
axs[2].set_xlabel("false positive rate") 
axs[3].set_xlabel("false positive rate")    
axs[0].set_ylabel("true positive rate") 
axs[2].set_ylabel("true positive rate");


#%%
### GMM BAYES CLASSIFIER #######################################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("ROC - GMM Bayes Classifier", fontsize='14', y=0.96)

gmmb = GMMBayes(n_components=6)   #choosen classifier
print("\n\n--- GMM Bayes Classifier ---")
for i in range(4):
    gmmb.fit(Xtrain[:,i][:, np.newaxis], ytrain)   #I train each color separately
    
    yprob = gmmb.predict_proba(Xtest[:,i][:, np.newaxis])
    ypred = gmmb.predict(Xtest[:,i][:, np.newaxis])
    
    fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #compute true and false positive rate
    
    axs[i].plot(fpr, tpr, lw=2, color='navy')   #plot the ROC curve
    axs[i].set_title(titles[i])
    
    print("\n", titles[i])
    print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
    print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
    print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly
    
if zoom:
    axs[0].set_ylim(min_zoom, 1)    
axs[2].set_xlabel("false positive rate") 
axs[3].set_xlabel("false positive rate")    
axs[0].set_ylabel("true positive rate") 
axs[2].set_ylabel("true positive rate");


#%%
### K-NEAREST NEIGHBOR CLASSIFIER ##############################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("ROC - K-Nearest Neighbor Classifier", fontsize='14', y=0.96)

knc = KNeighborsClassifier(n_neighbors=10)   #choosen classifier
print("\n\n--- K-Nearest Neighbor Classifier ---")
for i in range(4):
    knc.fit(Xtrain[:,i][:, np.newaxis], ytrain)   #I train each color separately
    
    yprob = knc.predict_proba(Xtest[:,i][:, np.newaxis])
    ypred = knc.predict(Xtest[:,i][:, np.newaxis])
    
    fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #compute true and false positive rate
    
    axs[i].plot(fpr, tpr, lw=2, color='navy')   #plot the ROC curve
    axs[i].set_title(titles[i])
    
    print("\n", titles[i])
    print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
    print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
    print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly

if zoom:
    axs[0].set_ylim(min_zoom, 1)    
axs[2].set_xlabel("false positive rate") 
axs[3].set_xlabel("false positive rate")    
axs[0].set_ylabel("true positive rate") 
axs[2].set_ylabel("true positive rate");


#%%
### GMMB WITH WHOLE DATASET ####################################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, ax = plt.subplots(1, figsize=(6,6))
fig.suptitle("ROC - GMM Bayes Classifier", fontsize='14', y=0.96)

print("\n\n-> What if I train on the whole dataset?")
print("\n--- GMM Bayes Classifier ---")
gmmb.fit(Xtrain, ytrain)   #fit on the whole dataset

yprob = gmmb.predict_proba(Xtest)
ypred = gmmb.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #compute true and false positive rate

ax.plot(fpr, tpr, lw=3, color='navy')   #plot the ROC curve
ax.set_title("Whole datset")
    
if zoom:
    ax.set_ylim(min_zoom, 1)    
ax.set_xlabel("false positive rate")    
ax.set_ylabel("true positive rate");

print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly 


#%%
### NAIVE BAYES WITH WHOLE DATASET #############################################################################

Xtrain, Xtest, ytrain, ytest = train_test_split(colors.T, source, test_size=0.30)

fig, ax = plt.subplots(1, figsize=(6,6))
fig.suptitle("ROC - Naive Bayes", fontsize='14', y=0.96)

print("\n--- Naive Bayes ---")
clf.fit(Xtrain, ytrain)   #fit on the whole dataset

yprob = clf.predict_proba(Xtest)
ypred = clf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #compute true and false positive rate

ax.plot(fpr, tpr, lw=3, color='navy')   #plot the ROC curve
ax.set_title("Whole datset")
    
if zoom:
    ax.set_ylim(min_zoom, 1)    
ax.set_xlabel("false positive rate")    
ax.set_ylabel("true positive rate");

print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly 

