from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

### DATA AND PARAMETERS ########################################################################################

urllib.request.urlretrieve("https://raw.githubusercontent.com/nshaud/ml_for_astro/main/stars.csv", "stars.csv")
df_stars = pd.read_csv("stars.csv")

le = LabelEncoder()
# Assign unique integers from 0 to 6 to each star type
df_stars['Star type'] = le.fit_transform(df_stars['Star type'])
labels = le.inverse_transform(df_stars['Star type'])

star_type = df_stars.loc[:, 'Star type']
star_type = np.array(star_type)

N = 240   #number of stars
print_k = False

#%%
### FUNCTIONS ##################################################################################################

def error(X, y, fit_method):
    Y = fit_method.predict(X)
    return np.sqrt( np.sum( (y-Y)**2 ) / len(X) )


#%%
### PCA ########################################################################################################

print("\n--- PCA ---")

### usual HR diagram ###
fig = plt.figure(figsize=(7, 7))
sns.scatterplot(data=df_stars, x='Temperature (K)', y='Luminosity(L/Lo)', hue=labels)
plt.xscale('log')
plt.yscale('log')
plt.title("HR diagram")
plt.xticks([5000, 10000, 50000])
plt.xlim(5e4, 1.5e3)
plt.show()


### new HR disgram ###
panda = df_stars.loc[:, 'Temperature (K)':'Absolute magnitude(Mv)']   #selecting data
panda = np.array(panda)
X = StandardScaler().fit_transform(panda)   #rescaling the data

pca = PCA(n_components=2)   #running the PCA
pca.fit(X)

proj = pca.transform(X)   #projection onto the new features
info = pca.explained_variance_ratio_   #information of each new feature
comp = pca.components_   #which old component is prevalent

names = ['Temperature', 'Luminosity', 'Radius', 'Absolute Magnitude']
for i in range(2):
    print("\n- PCA"+str(i+1)+" contains", np.around(info[i]*100, 1), "% of the information")
    print("  and it's composed by: ")
    sum_comp = np.sum(np.abs(comp[i]))
    for j in range(len(names)):
          print("  ", str(names[j])+":", np.around(np.abs(comp[i, j]/sum_comp)*100, 1), "%")  
print("\nWith 2 components I obtain", np.around(np.sum(info)*100, 1), "% of the information")

fig = plt.figure(figsize=(7, 7))
sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=labels)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("HR diagram with PCA")
plt.show()


#%%
### K FOLDING ##################################################################################################

max_dephts = np.arange(1, 11)   #range of max_dephts to test
K = 20   #number of K-folds

TrainErr = []
CvErr = []
if print_k:
    print("\n--- Kfold ---")
kf = KFold(n_splits=K, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    if print_k:
        print("Fold:", i)
        print("  Train:", train_index)
        print("  Test:", test_index)

    Xtrain = X[train_index]
    Xtest = X[test_index]
    ytrain = star_type[train_index]
    ytest = star_type[[test_index]]

    te = []
    cve = []
    for md in max_dephts:
        clf = DecisionTreeClassifier(max_depth=md)
        clf.fit(Xtrain, ytrain)

        te.append(error(Xtrain, ytrain, clf))   #computing the training error on this fold
        cve.append(error(Xtest, ytest, clf))   #computing the cross validation error on this fold
    
    TrainErr.append(np.array(te))
    CvErr.append(np.array(cve))

TrainErr_med = np.median(TrainErr, axis=0)
CvErr_med = np.median(CvErr, axis=0)

plt.figure()
plt.plot(max_dephts, TrainErr_med, color='navy', label="Training err")
plt.plot(max_dephts, CvErr_med, color='navy', ls='dashed', label="CV error")
plt.xlabel('maximum depht')
plt.xlim(0,10)
plt.title("Cross validation - Kfold")
plt.legend()
plt.show()


#%%
### DECISION TREE ##############################################################################################

print("\n--- Decision Tree ---")

#dividing in test and cv
Xtrain, Xtest, ytrain, ytest = train_test_split(proj, star_type, test_size=0.30)

#chose max depth
MaxD_mr = 5

#computing the efficiency
clf = DecisionTreeClassifier(max_depth=MaxD_mr, criterion='entropy')
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)

C = confusion_matrix(ytest, ypred)
print("\nEfficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.imshow(C, cmap='Blues', interpolation='nearest')
plt.yticks(np.arange(6), le.classes_, fontsize=8)
plt.xticks(np.arange(6), le.classes_, rotation=45, fontsize=8)
plt.title("Confusion matrix")
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()

fig, ax = plt.subplots(1, figsize=(7, 7))
DecisionBoundaryDisplay.from_estimator(clf, proj, response_method='predict', cmap='rainbow', alpha=0.3, ax=ax)
ax.scatter(proj[:,0], proj[:,1], c=star_type, cmap='rainbow', edgecolors='k')
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("HR diagram with PCA");


#%%
### DECISION TREE WITHOUT PCA ##################################################################################

print("\n--- Decision Tree without PCA ---")
print("\nUsing just rescaled data")

#dividing in test and cv
Xtrain, Xtest, ytrain, ytest = train_test_split(X, star_type, test_size=0.30)

#chose max depth
MaxD_mr = 5

#computing the efficiency
clf = DecisionTreeClassifier(max_depth=MaxD_mr, criterion='entropy')
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)

C = confusion_matrix(ytest, ypred)
print("Efficiency:", np.round((np.sum(C.diagonal())/len(ytest))*100, 2), "%")

plt.figure()
plt.imshow(C, cmap='Blues', interpolation='nearest')
plt.yticks(np.arange(6), le.classes_, fontsize=8)
plt.xticks(np.arange(6), le.classes_, rotation=45, fontsize=8)
plt.title("Confusion matrix - without PCA")
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()