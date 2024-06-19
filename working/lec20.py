import numpy as np
import matplotlib.pyplot as plt
import h5py
import corner
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from tqdm import tqdm

################################################################################################################

conceal_warnings = True

if conceal_warnings:
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    
#%%
### DATA #######################################################################################################

data = h5py.File("../../sample_2e7_design_precessing_higherordermodes_3detectors.h5", "r")
#print(list(data.keys()))
N = len(data['q'])

corner_plot = True


#%%
### FUNCTIONS ##################################################################################################
    
def plot_contour(method, xmin, xmax, ymin, ymax):
    
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 50), np.linspace(ymin, ymax, 50))
    xystack = np.vstack([xx.ravel(),yy.ravel()])
    Xgrid = xystack.T

    Z = method.predict(Xgrid)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, [0.5], colors='k', linewidths=2)
    
    
def plot_compl_cont(method, Xtrain, Xtest, ytrain, ytest):
    completeness_test = []
    contamination_test = []
    completeness_train = []
    contamination_train = []
    
    for md in tqdm(drange):
        clf = method(max_depth=md)
        clf.fit(Xtrain, ytrain)   #fit every train set with different maximum depth

        #completeness and contamination of test set
        ypred_test = clf.predict(Xtest)
        completeness_test.append( metrics.recall_score(ytest, ypred_test) )
        contamination_test.append( 1-metrics.precision_score(ytest, ypred_test) )
     
        #completeness and contamination of train set
        ypred_train = clf.predict(Xtrain)
        completeness_train.append( metrics.recall_score(ytrain, ypred_train) )
        contamination_train.append( 1-metrics.precision_score(ytrain, ypred_train) )
        
    plt.figure()
    plt.plot(drange,completeness_test,label='completeness test', color='orange')
    plt.plot(drange,completeness_train,label='completeness train', color='orange', ls='dotted')
    plt.plot(drange,contamination_test,label='contamination test', color='coral')
    plt.plot(drange,contamination_train,label='contamination train', color='coral', ls='dotted')
    plt.xticks(np.arange(min(drange), max(drange)+1, 1.0))
    plt.title("Completeness and contamination")
    plt.legend()
    
    
#%%
### DOWNSAMPLING ###############################################################################################

### downsampling ###
step = 8000   #with smaller steps the running time is much longer, but results aren't much better
ind = np.arange(0, N, step)   #selected indices
print("\nIn this analysis I will use", len(ind), "datapoints")

#organize the downsampled dataset
snr = data['det'][ind]
features = np.array([data['mtot'][ind],
                     data['q'][ind],
                     data['iota'][ind],
                     data['psi'][ind],
                     data['z'][ind],
                     data['dec'][ind],
                     data['ra'][ind],
                     data['chi1x'][ind],
                     data['chi1y'][ind],
                     data['chi1z'][ind],
                     data['chi2x'][ind], 
                     data['chi2y'][ind],
                     data['chi2z'][ind]])
labels = ['mtot', 'q', 'i', 'psi', 'z', 'dec', 'ra',
          'spin 1x', 'spin 1y', 'spin 1z', 'spin 2x', 'spin 2y', 'spin 2z']

#plotting all the features
if corner_plot:
    fig = corner.corner(features.T, labels=labels, levels=[0.68,0.95], color='navy')
    fig.suptitle("Raw data downsampled");

#selecting the indices with SNR > 12
s_good = (ind[np.where(snr==1)] / step).astype(int)
s_bad = (ind[np.where(snr==0)] / step).astype(int)


#%%
### ALL DATA ###################################################################################################

print("\n\n--- ALL DATA ---")

X = (StandardScaler().fit_transform(features)).T  #rescale the data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, snr, test_size=0.30)


#%%
### DECISION TREE ##############################################################################################

print("\n- Decision Tree -")

### cross validation ###
clf = DecisionTreeClassifier()
drange = np.arange(1, 21)

grid = GridSearchCV(clf, param_grid={'max_depth': drange}, cv=5, n_jobs=-1)
grid.fit(X, snr)
MD_dt = grid.best_params_['max_depth']   #best maximum depth

print("\nFrom cross validation the best maximum depht is", MD_dt)
if MD_dt == drange[-1]:
    print("Warning:", drange[-1], "is the maximum max_depht between the tried ones")

#plot completeness and contamination while varying the maximum depth
plot_compl_cont(DecisionTreeClassifier, Xtrain, Xtest, ytrain, ytest)


### classification ###
clf = DecisionTreeClassifier(max_depth=MD_dt)   #decision tree with the best maximum depth
clf.fit(Xtrain, ytrain)

yprob = clf.predict_proba(Xtest)
ypred = clf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #false positives and true positives ratios

#plot the ROC curve
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Decision Tree - all data")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()

print("\nGoodness of the classification")
print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly


#%%
### RANDOM FOREST ##############################################################################################

print("\n- Random Forest -")

### cross validation ###
rf = RandomForestClassifier()
drange = np.arange(1, 21)

grid = GridSearchCV(rf, param_grid={'max_depth': drange}, cv=5, n_jobs=-1)
grid.fit(X, snr)
MD_rf = grid.best_params_['max_depth']   #best maximum depth

print("\nFrom cross validation the best maximum depht is", MD_rf)
if MD_rf == drange[-1]:
    print("Warning:", drange[-1], "is the maximum max_depht between the tried ones")

#plot completeness and contamination while varying the maximum depth
plot_compl_cont(RandomForestClassifier, Xtrain, Xtest, ytrain, ytest)


### classification ###
rf = RandomForestClassifier(max_depth=MD_rf)   #random forest with the best maximum depth
rf.fit(Xtrain, ytrain)

yprob = rf.predict_proba(Xtest)
ypred = rf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #false positives and true positives ratios

#plot the ROC curve
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Random Forest - all data")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()

print("\nGoodness of the classification")
print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly


#%%
### MASS AND REDSHIFT ##########################################################################################

print("\n\n--- MASS AND REDSHIFT ---")

mz = np.array([features[0], features[4]]).T
X = (StandardScaler().fit_transform(mz))   #rescale the data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, snr, test_size=0.30)


#%%
### DECISION TREE ##############################################################################################

print("\n- Decision Tree -")

### cross validation ###
clf = DecisionTreeClassifier()
drange = np.arange(1, 21)

grid = GridSearchCV(clf, param_grid={'max_depth': drange}, cv=5, n_jobs=-1)
grid.fit(X, snr)
MD_dt = grid.best_params_['max_depth']   #best maximum depth

print("\nFrom cross validation the best maximum depht is", MD_dt)
if MD_dt == drange[-1]:
    print("Warning:", drange[-1], "is the maximum max_depht between the tried ones")

#plot completeness and contamination while varying the maximum depth
plot_compl_cont(DecisionTreeClassifier, Xtrain, Xtest, ytrain, ytest)


### classification ###
clf = DecisionTreeClassifier(max_depth=MD_dt)   #decision tree with the best maximum depth
clf.fit(Xtrain, ytrain)

yprob = clf.predict_proba(Xtest)
ypred = clf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #false positives and true positives ratios

#plot the ROC curve
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Decision Tree - mass and redshift")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()

print("\nGoodness of the classification")
print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly


#%%
### RANDOM FOREST ##############################################################################################

print("\n- Random Forest -")

### cross validation ###
rf = RandomForestClassifier()
drange = np.arange(1, 21)

grid = GridSearchCV(rf, param_grid={'max_depth': drange}, cv=5, n_jobs=-1)
grid.fit(X, snr)
MD_rf = grid.best_params_['max_depth']   #best maximum depth

print("\nFrom cross validation the best maximum depht is", MD_rf)
if MD_rf == drange[-1]:
    print("Warning:", drange[-1], "is the maximum max_depht between the tried ones")

#plot completeness and contamination while varying the maximum depth
plot_compl_cont(RandomForestClassifier, Xtrain, Xtest, ytrain, ytest)


### classification ###
rf = RandomForestClassifier(max_depth=MD_rf)   #random forest with the best maximum depth
rf.fit(Xtrain, ytrain)

yprob = rf.predict_proba(Xtest)
ypred = rf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #false positives and true positives ratios

#plot the ROC curve
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Random Forest - mass and redshift")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()

print("\nGoodness of the classification")
print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly


### visualize on the whole set ###
snr_pred = rf.predict(X)

s_good_pred = (ind[np.where(snr_pred==1)] / step).astype(int)
s_bad_pred = (ind[np.where(snr_pred==0)] / step).astype(int)

plt.figure()
plt.scatter(X[:,0][s_bad], X[:,1][s_bad], marker='.', color='royalblue', s=10, label="SNR < 12")
plt.scatter(X[:,0][s_good], X[:,1][s_good], marker='.', color='crimson', s=10, label="SNR > 12")
plot_contour(rf, -2, 2, -2, 2)
plt.xlim(min(X[:,0]), max(X[:,0]))
plt.ylim(min(X[:,1]), max(X[:,1]))
plt.xlabel("$m_{tot}$ (rescaled)")
plt.ylabel("z (rescaled)")
plt.title("Classification with mass and redshift")
plt.legend(loc='upper right')
plt.show()


#%%
### PCA #######################################################################################################

print("\n\n--- PCA ---")

ndim = 4   #I want at least 90% of the information
print("I will use", ndim, "components")

features_scaled = (StandardScaler().fit_transform(features)).T   #rescale the data

pca = PCA(n_components=ndim)   #running the PCA
pca.fit(features_scaled)

proj = pca.transform(features_scaled)   #projection onto the new features
info = pca.explained_variance_ratio_   #information of each new feature
composition = pca.components_   #which old component is prevalent

# show results
for i in range(ndim):
    print("\n- PCA"+str(i+1)+" contains", np.around(info[i]*100, 1), "% of the information")
    print("  and it's composed by: ")
    sum_comp = np.sum(np.abs(composition[i]))
    for j in range(len(labels)):
          print("  ", str(labels[j])+":", np.around(np.abs(composition[i, j]/sum_comp)*100, 1), "%") 
print("\nWith", ndim, "components I obtain", np.around(np.sum(info)*100, 1), "% of the information")

Xtrain, Xtest, ytrain, ytest = train_test_split(proj, snr, test_size=0.30)


#%%
### DECISION TREE ##############################################################################################

print("\n- Decision Tree -")

### cross validation ###
clf = DecisionTreeClassifier()
drange = np.arange(1, 21)

grid = GridSearchCV(clf, param_grid={'max_depth': drange}, cv=5, n_jobs=-1)
grid.fit(X, snr)
MD_dt = grid.best_params_['max_depth']   #best maximum depth

print("\nFrom cross validation the best maximum depht is", MD_dt)
if MD_dt == drange[-1]:
    print("Warning:", drange[-1], "is the maximum max_depht between the tried ones")

#plot completeness and contamination while varying the maximum depth
plot_compl_cont(DecisionTreeClassifier, Xtrain, Xtest, ytrain, ytest)


### classification ###
clf = DecisionTreeClassifier(max_depth=MD_dt)   #decision tree with the best maximum depth
clf.fit(Xtrain, ytrain)

yprob = clf.predict_proba(Xtest)
ypred = clf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #false positives and true positives ratios

#plot the ROC curve
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Decision Tree - PCA")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()

print("\nGoodness of the classification")
print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly


#%%
### RANDOM FOREST ##############################################################################################

print("\n- Random Forest -")

### cross validation ###
rf = RandomForestClassifier()
drange = np.arange(1, 21)

grid = GridSearchCV(rf, param_grid={'max_depth': drange}, cv=5, n_jobs=-1)
grid.fit(X, snr)
MD_rf = grid.best_params_['max_depth']   #best maximum depth

print("\nFrom cross validation the best maximum depht is", MD_rf)
if MD_rf == drange[-1]:
    print("Warning:", drange[-1], "is the maximum max_depht between the tried ones")

#plot completeness and contamination while varying the maximum depth
plot_compl_cont(RandomForestClassifier, Xtrain, Xtest, ytrain, ytest)


### classification ###
rf = RandomForestClassifier(max_depth=MD_rf)   #random forest with the best maximum depth
rf.fit(Xtrain, ytrain)

yprob = rf.predict_proba(Xtest)
ypred = rf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])   #false positives and true positives ratios

#plot the ROC curve
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Random Forest - PCA")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()

print("\nGoodness of the classification")
print("Completeness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly


