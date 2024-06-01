import numpy as np
import matplotlib.pyplot as plt
import h5py
import corner
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


#%%
### DATA #######################################################################################################

data = h5py.File("../../sample_2e7_design_precessing_higherordermodes_3detectors.h5", "r")
#print(list(data.keys()))
N = len(data['q'])

corner_plot = False


#%%
### FUNCTIONS ##################################################################################################
    
def plot_contour(method, xmin, xmax, ymin, ymax):
    
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 50), np.linspace(ymin, ymax, 50))
    xystack = np.vstack([xx.ravel(),yy.ravel()])
    Xgrid = xystack.T

    Z = method.predict(Xgrid)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, [0.5], colors='k', linewidths=2)
    
#%%
### DOWNSAMPLING ###############################################################################################

### downsampling ###
step = 8000
ind = np.arange(0, N, step)   #selected indices
print("\nIn this analysis I will use", len(ind), "datapoints")

#organize the downsampled dataset
snr = data['det'][ind]
features = np.array([data['mtot'][ind],
                     data['q'][ind],
                     data['iota'][ind],
                     data['psi'][ind],
                     data['z'][ind],
                     data['chi1x'][ind],
                     data['chi1y'][ind],
                     data['chi1z'][ind],
                     data['chi2x'][ind], 
                     data['chi2y'][ind],
                     data['chi2z'][ind]])
labels = ['mtot', 'q', 'i', 'psi', 'z', 'spin 1x', 'spin 1y', 'spin 1z', 'spin 2x', 'spin 2y', 'spin 2z']

#plotting all the features
if corner_plot:
    fig = corner.corner(features.T, labels=labels, levels=[0.68,0.95], color='navy')
    fig.suptitle("Raw data downsampled");

#selecting the indices with SNR > 12
s_good = (ind[np.where(snr==1)] / step).astype(int)
s_bad = (ind[np.where(snr==0)] / step).astype(int)


#%%
################################################################################################################
### DT - ALL DATA ##############################################################################################

print("\n\n--- ALL DATA ---")
print("\n- Decision Tree -")

#dividing in test and cv
X = (StandardScaler().fit_transform(features)).T  #rescale the data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, snr, test_size=0.30)

completeness_test = []
contamination_test = []
completeness_train = []
contamination_train = []

print("\nChoosing maximum depht:")
max_dephts = np.arange(1, 20)
for md in tqdm(max_dephts):
    clf = DecisionTreeClassifier(max_depth=md, criterion='entropy')
    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xtest)
    
    C = confusion_matrix(ytest, ypred)
    tn, fp, fn, tp = C.ravel()
    completeness_test.append(tp/(tp+fn))
    contamination_test.append(fp/(tp+fp))

    ypred = clf.predict(Xtrain)
    C = confusion_matrix(ytrain, ypred)
    tn, fp, fn, tp = C.ravel()
    completeness_train.append(tp/(tp+fn))
    contamination_train.append(fp/(tp+fp))
    
plt.figure()
plt.plot(max_dephts,completeness_test,label='completeness test', color='orange')
plt.plot(max_dephts,completeness_train,label='completeness train', color='orange', ls='dotted')
plt.plot(max_dephts,contamination_test,label='contamination test', color='coral')
plt.plot(max_dephts,contamination_train,label='contamination train', color='coral', ls='dotted')
plt.title("Completeness and contamination")
plt.legend()

#chose max depth
MaxD_mr = 4

### ROC curve ###
clf = DecisionTreeClassifier(max_depth=MaxD_mr, criterion='entropy')
clf.fit(Xtrain, ytrain)

yprob = clf.predict_proba(Xtest)
ypred = clf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])

C = confusion_matrix(ytest, ypred)
tn, fp, fn, tp = C.ravel()
print("\n\nGoodness of the classification:")
print("False positive rate:", np.round((fp/(tp+fp))*100, 2), "%")
print("False negative rate:", np.round((fn/(tn+fn))*100, 2), "%")
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Decision Tree")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()


#%%
### RF - ALL DATA ##############################################################################################

print("\n- Random Forest -")

### ROC curve ###
rf = RandomForestClassifier(max_depth=MaxD_mr)
rf.fit(Xtrain, ytrain)

yprob = rf.predict_proba(Xtest)
ypred = rf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])

C = confusion_matrix(ytest, ypred)
tn, fp, fn, tp = C.ravel()
print("\nGoodness of the classification:")
print("False positive rate:", np.round((fp/(tp+fp))*100, 2), "%")
print("False negative rate:", np.round((fn/(tn+fn))*100, 2), "%")
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Random Forest")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()


#%%
################################################################################################################
### DT - MASS AND REDSHIFT #####################################################################################

print("\n\n--- MASS AND REDSHIFT ---")
print("\n- Decision Tree -")

#dividing in test and cv
mz = np.array([features[0], features[4]]).T
X = (StandardScaler().fit_transform(mz))   #rescale the data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, snr, test_size=0.30)

completeness_test = []
contamination_test = []
completeness_train = []
contamination_train = []

print("\nChoosing maximum depht:")
max_dephts = np.arange(1, 20)
for md in tqdm(max_dephts):
    clf = DecisionTreeClassifier(max_depth=md, criterion='entropy')
    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xtest)
    
    C = confusion_matrix(ytest, ypred)
    tn, fp, fn, tp = C.ravel()
    completeness_test.append(tp/(tp+fn))
    contamination_test.append(fp/(tp+fp))

    ypred = clf.predict(Xtrain)
    C = confusion_matrix(ytrain, ypred)
    tn, fp, fn, tp = C.ravel()
    completeness_train.append(tp/(tp+fn))
    contamination_train.append(fp/(tp+fp))
    
plt.figure()
plt.plot(max_dephts,completeness_test,label='completeness test', color='orange')
plt.plot(max_dephts,completeness_train,label='completeness train', color='orange', ls='dotted')
plt.plot(max_dephts,contamination_test,label='contamination test', color='coral')
plt.plot(max_dephts,contamination_train,label='contamination train', color='coral', ls='dotted')
plt.title("Completeness and contamination - mass and redshift")
plt.legend()

#chose max depth
MaxD_mr = 6

### ROC curve ###
clf = DecisionTreeClassifier(max_depth=MaxD_mr, criterion='entropy')
clf.fit(Xtrain, ytrain)

yprob = clf.predict_proba(Xtest)
ypred = clf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])

C = confusion_matrix(ytest, ypred)
tn, fp, fn, tp = C.ravel()
print("\n\nGoodness of the classification:")
print("False positive rate:", np.round((fp/(tp+fp))*100, 2), "%")
print("False negative rate:", np.round((fn/(tn+fn))*100, 2), "%")
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Decision Tree - mass and redshift")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()


#%%
### RF - MASS AND REDSHIFT #####################################################################################

print("\n- Random Forest -")

### ROC curve ###
rf = RandomForestClassifier(max_depth=MaxD_mr)
rf.fit(Xtrain, ytrain)

yprob = rf.predict_proba(Xtest)
ypred = rf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])

C = confusion_matrix(ytest, ypred)
tn, fp, fn, tp = C.ravel()
print("\nGoodness of the classification:")
print("False positive rate:", np.round((fp/(tp+fp))*100, 2), "%")
print("False negative rate:", np.round((fn/(tn+fn))*100, 2), "%")
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Random Forest - mass and redshift")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()


### whole set ###
rf = RandomForestClassifier(max_depth=3)
rf.fit(X, snr)
snr_pred = rf.predict(X)

s_good_pred = (ind[np.where(snr_pred==1)] / step).astype(int)
s_bad_pred = (ind[np.where(snr_pred==0)] / step).astype(int)

plt.figure()
plt.scatter(X[:,0][s_bad], X[:,1][s_bad], marker='.', color='royalblue', s=10, label="SNR < 12")
plt.scatter(X[:,0][s_good], X[:,1][s_good], marker='.', color='crimson', s=10, label="SNR > 12")
plot_contour(rf, -1.5, 1.5, -1.5, 1.5)
plt.xlabel("$m_{tot}$")
plt.ylabel("redshift")
plt.title("Results - mass and redshift")
plt.legend(loc='upper right')
plt.show()


#%%
################################################################################################################
### PCA3 #######################################################################################################

print("\n\n--- 3 DIM PCA ---")

features_scaled = (StandardScaler().fit_transform(features)).T   #rescale the data

### PCA ###
pca = PCA(n_components=3)   #running the PCA
pca.fit(features_scaled)

proj = pca.transform(features_scaled)   #projection onto the new features
info = pca.explained_variance_ratio_   #information of each new feature
composition = pca.components_   #which old component is prevalent


### show results###
for i in range(3):
    print("\n- PCA"+str(i+1)+" contains", np.around(info[i]*100, 1), "% of the information")
    print("  and it's composed by: ")
    sum_comp = np.sum(np.abs(composition[i]))
    for j in range(len(labels)):
          print("  ", str(labels[j])+":", np.around(np.abs(composition[i, j]/sum_comp)*100, 1), "%") 
print("\nWith 3 components I obtain", np.around(np.sum(info)*100, 1), "% of the information")

#3D plot
ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
ax.scatter(proj[:,0][s_bad], proj[:,1][s_bad], proj[:,2][s_bad], label="SNR < 12",
           color='royalblue', s=5, alpha=0.5)
ax.scatter(proj[:,0], proj[:,1], proj[:,2], c=snr,
           cmap='coolwarm', s=5, alpha=0.5)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
ax.set_title("3d PCA processed points")
plt.show()

#2D projectionts plot
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()
fig.suptitle("3D projections of PCA points")
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


#%%
### DT - PCA3 ##################################################################################################

print("\n- Decision Tree -")

#dividing in test and cv
Xtrain, Xtest, ytrain, ytest = train_test_split(proj, snr, test_size=0.30)

completeness_test = []
contamination_test = []
completeness_train = []
contamination_train = []

print("\nChoosing maximum depht:")
max_dephts = np.arange(2, 20)
for md in tqdm(max_dephts):
    clf = DecisionTreeClassifier(max_depth=md, criterion='entropy')
    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xtest)
    
    C = confusion_matrix(ytest, ypred)
    tn, fp, fn, tp = C.ravel()
    completeness_test.append(tp/(tp+fn))
    contamination_test.append(fp/(tp+fp))

    ypred = clf.predict(Xtrain)
    C = confusion_matrix(ytrain, ypred)
    tn, fp, fn, tp = C.ravel()
    completeness_train.append(tp/(tp+fn))
    contamination_train.append(fp/(tp+fp))
    
plt.figure()
plt.plot(max_dephts,completeness_test,label='completeness test', color='orange')
plt.plot(max_dephts,completeness_train,label='completeness train', color='orange', ls='dotted')
plt.plot(max_dephts,contamination_test,label='contamination test', color='coral')
plt.plot(max_dephts,contamination_train,label='contamination train', color='coral', ls='dotted')
plt.title("Completeness and contamination - 3d PCA")
plt.legend()

#chose max depth
MaxD_pca = 6


### ROC curve ###
clf = DecisionTreeClassifier(max_depth=MaxD_pca, criterion='entropy')
clf.fit(Xtrain, ytrain)

yprob = clf.predict_proba(Xtest)
ypred = clf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])

C = confusion_matrix(ytest, ypred)
tn, fp, fn, tp = C.ravel()
print("\n\nGoodness of the classification:")
print("False positive rate:", np.round((fp/(tp+fp))*100, 2), "%")
print("False negative rate:", np.round((fn/(tn+fn))*100, 2), "%")
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Decision Tree - 3d PCA")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()


#%%
### RF - PCA3 ##################################################################################################

print("\n- Random Forest -")

### ROC curve ###
Xtrain, Xtest, ytrain, ytest = train_test_split(proj, snr, test_size=0.30)

rf = RandomForestClassifier(max_depth=MaxD_pca)
rf.fit(Xtrain, ytrain)

yprob = rf.predict_proba(Xtest)
ypred = rf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])

C = confusion_matrix(ytest, ypred)
tn, fp, fn, tp = C.ravel()
print("\nGoodness of the classification:")
print("False positive rate:", np.round((fp/(tp+fp))*100, 2), "%")
print("False negative rate:", np.round((fn/(tn+fn))*100, 2), "%")
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Random Forest - 3d PCA")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()


### whole set ###
rf = RandomForestClassifier(max_depth=MaxD_pca)
rf.fit(proj, snr)
snr_pred = rf.predict(proj)

s_good_pred = (ind[np.where(snr_pred==1)] / step).astype(int)
s_bad_pred = (ind[np.where(snr_pred==0)] / step).astype(int)


#2D projectionts plot
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()
fig.suptitle("Results - 3d PCA")
axs[0].scatter(proj[:,0][s_bad], proj[:,1][s_bad], color='royalblue', label="SNR < 12", s=50, alpha=0.3)
axs[0].scatter(proj[:,0][s_good], proj[:,1][s_good], color='crimson', label="SNR > 12", s=50, alpha=0.3)
axs[0].scatter(proj[:,0][s_bad_pred], proj[:,1][s_bad_pred], marker='x', color='royalblue', s=15, label="SNR < 12")
axs[0].scatter(proj[:,0][s_good_pred], proj[:,1][s_good_pred], marker='x', color='crimson', s=15, label="SNR > 12")
#axs[0].set_xscale('log')
#axs[0].set_yscale('log')
axs[0].set_xlabel("PCA1")
axs[0].set_ylabel("PCA2")

axs[1].scatter(proj[:,1][s_bad], proj[:,2][s_bad], color='royalblue', label="SNR < 12", s=50, alpha=0.3)
axs[1].scatter(proj[:,1][s_good], proj[:,2][s_good], color='crimson', label="SNR > 12", s=50, alpha=0.3)
axs[1].scatter(proj[:,1][s_bad_pred], proj[:,2][s_bad_pred], marker='x', color='royalblue', s=15, label="SNR < 12")
axs[1].scatter(proj[:,1][s_good_pred], proj[:,2][s_good_pred], marker='x', color='crimson', s=15, label="SNR > 12")
#axs[1].set_xscale('log')
#axs[1].set_yscale('log')
axs[1].set_xlabel("PCA2")
axs[1].set_ylabel("PCA3")

axs[2].scatter(proj[:,2][s_bad], proj[:,0][s_bad], color='royalblue', label="SNR < 12", s=50, alpha=0.3)
axs[2].scatter(proj[:,2][s_good], proj[:,0][s_good], color='crimson', label="SNR > 12", s=50, alpha=0.3)
axs[2].scatter(proj[:,2][s_bad_pred], proj[:,0][s_bad_pred], marker='x', color='royalblue', s=15, label="SNR < 12")
axs[2].scatter(proj[:,2][s_good_pred], proj[:,0][s_good_pred], marker='x', color='crimson', s=15, label="SNR > 12")
#axs[2].set_xscale('log')
#axs[2].set_yscale('log')
axs[2].set_xlabel("PCA3")
axs[2].set_ylabel("PCA1");


#%%
################################################################################################################
### PCA2 #######################################################################################################

print("\n\n--- 2 DIM PCA ---")

features_scaled = (StandardScaler().fit_transform(features)).T   #rescale the data

### PCA ###
pca = PCA(n_components=2)   #running the PCA
pca.fit(features_scaled)

proj = pca.transform(features_scaled)   #projection onto the new features
info = pca.explained_variance_ratio_   #information of each new feature
composition = pca.components_   #which old component is prevalent


### show results###
for i in range(2):
    print("\n- PCA"+str(i+1)+" contains", np.around(info[i]*100, 1), "% of the information")
    print("  and it's composed by: ")
    sum_comp = np.sum(np.abs(composition[i]))
    for j in range(len(labels)):
          print("  ", str(labels[j])+":", np.around(np.abs(composition[i, j]/sum_comp)*100, 1), "%")  
print("\nWith 2 components I obtain", np.around(np.sum(info)*100, 1), "% of the information")

#2D projectionts plot
plt.figure()
plt.scatter(proj[:,0][s_bad], proj[:,1][s_bad], color='royalblue', label="SNR < 12", alpha=0.5)
plt.scatter(proj[:,0][s_good], proj[:,1][s_good], color='crimson', label="SNR > 12", alpha=0.5)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("2d PCA processed points")
plt.show()


#%%
### DT - PCA2 ##################################################################################################

print("\n- Decision Tree -")

#dividing in test and cv
Xtrain, Xtest, ytrain, ytest = train_test_split(proj, snr, test_size=0.30)

completeness_test = []
contamination_test = []
completeness_train = []
contamination_train = []

print("\nChoosing maximum depht:")
max_dephts = np.arange(2, 20)
for md in tqdm(max_dephts):
    clf = DecisionTreeClassifier(max_depth=md, criterion='entropy')
    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xtest)
    
    C = confusion_matrix(ytest, ypred)
    tn, fp, fn, tp = C.ravel()
    completeness_test.append(tp/(tp+fn))
    contamination_test.append(fp/(tp+fp))

    ypred = clf.predict(Xtrain)
    C = confusion_matrix(ytrain, ypred)
    tn, fp, fn, tp = C.ravel()
    completeness_train.append(tp/(tp+fn))
    contamination_train.append(fp/(tp+fp))
    
plt.figure()
plt.plot(max_dephts,completeness_test,label='completeness test', color='orange')
plt.plot(max_dephts,completeness_train,label='completeness train', color='orange', ls='dotted')
plt.plot(max_dephts,contamination_test,label='contamination test', color='coral')
plt.plot(max_dephts,contamination_train,label='contamination train', color='coral', ls='dotted')
plt.title("Completeness and contamination - 2d PCA")
plt.legend()

#chose max depth
MaxD_pca = 5


### ROC curve ###
clf = DecisionTreeClassifier(max_depth=MaxD_pca, criterion='entropy')
clf.fit(Xtrain, ytrain)

yprob = clf.predict_proba(Xtest)
ypred = clf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])

C = confusion_matrix(ytest, ypred)
tn, fp, fn, tp = C.ravel()
print("\n\nGoodness of the classification:")
print("False positive rate:", np.round((fp/(tp+fp))*100, 2), "%")
print("False negative rate:", np.round((fn/(tn+fn))*100, 2), "%")
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Decision Tree - 2d PCA")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()


#%%
### RF - PCA2 ##################################################################################################

print("\n- Random Forest -")

### ROC curve ###
Xtrain, Xtest, ytrain, ytest = train_test_split(proj, snr, test_size=0.30)

rf = RandomForestClassifier(max_depth=MaxD_pca)
rf.fit(Xtrain, ytrain)

yprob = rf.predict_proba(Xtest)
ypred = rf.predict(Xtest)

fpr, tpr, thresh = roc_curve(ytest, yprob[:,1])

C = confusion_matrix(ytest, ypred)
tn, fp, fn, tp = C.ravel()
print("\nGoodness of the classification:")
print("False positive rate:", np.round((fp/(tp+fp))*100, 2), "%")
print("False negative rate:", np.round((fn/(tn+fn))*100, 2), "%")
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.plot(fpr, tpr, lw=2, color='navy')
plt.title("Random Forest - 2d PCA")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()


### whole set ###
rf = RandomForestClassifier(max_depth=MaxD_pca)
rf.fit(proj, snr)
snr_pred = rf.predict(proj)

s_good_pred = (ind[np.where(snr_pred==1)] / step).astype(int)
s_bad_pred = (ind[np.where(snr_pred==0)] / step).astype(int)


#2D projectionts plot
plt.figure()
#plt.scatter(proj[:,0][s_bad], proj[:,1][s_bad], color='royalblue', s=50, alpha=0.3)
#plt.scatter(proj[:,0][s_good], proj[:,1][s_good], color='crimson', s=50, alpha=0.3)
plt.scatter(proj[:,0][s_bad_pred], proj[:,1][s_bad_pred], marker='.', color='royalblue', s=20, label="SNR < 12")
plt.scatter(proj[:,0][s_good_pred], proj[:,1][s_good_pred], marker='.', color='crimson', s=20, label="SNR > 12")
plot_contour(rf, -0.5, 3, -1.5, 2)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Results - 2d PCA")
plt.show()

