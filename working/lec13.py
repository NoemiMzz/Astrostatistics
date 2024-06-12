import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

### FUNCTIONS ##################################################################################################

def plot_embedding(X, title):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)

    for digit in digits.target_names:
        ax.scatter(
            *X[target == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")

### DATA AND PARAMETERS ########################################################################################

digits = load_digits()   #load the data
print("\nImport data:")
print(digits.images.shape)
print("")
print(digits.keys())

data, target = digits.data, digits.target   #organize samples and targets

plot_by_me = True

#%%
### UNSUPERVISED ###############################################################################################

### isomap ### 
embedding = Isomap(n_components=2, n_neighbors=7)
data_red = embedding.fit_transform(data)   #reduce dimensionality of data
print("\nReduced dimensions:")
print(data_red.shape)

if plot_by_me:
    plt.figure()
    plt.scatter(data_red[:,0], data_red[:,1], c=target, s=10, alpha=0.5, cmap='tab10')
    plt.colorbar(label='digit label', ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title("ISOMAP")
    plt.show()
else:
    plot_embedding(data_red, "ISOMAP")
    
    
#%%
### SUPERVISED #################################################################################################

### logistic regression ###
Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2)   #split in train and validation

#training the data
clf = LogisticRegression(solver='sag', max_iter=3000).fit(Xtrain, ytrain)

#computing predictions
pred_train = clf.predict(Xtrain)   #train set
pred_test = clf.predict(Xtest)   #test set     


### accuracy ###
#visualize errors in the validation set
print("\nValidation set:")
print("pred\t true")
for i in range(len(Xtest)):
    if pred_test[i] == ytest[i]:
        print(pred_test[i], "\t", ytest[i])
    else:
        print(pred_test[i], "\t", ytest[i], "  <--")   
        
print("\n\nAccuracy of training digits:")   #computing the accuracy score
print(accuracy_score(ytrain, pred_train))   #train set
print("\nAccuracy of test digits:")
print(accuracy_score(ytest, pred_test))   #test set

print("\nConfusion matrix of training digits:")   #computing the confusion matrix
print(confusion_matrix(ytrain, pred_train))   #train set
print("\nConfusion matrix of test digits:")
print(confusion_matrix(ytest, pred_test))   #test set

#visualize test confusion matrix
plt.figure()
plt.imshow(confusion_matrix(ytest, pred_test), cmap='Blues', interpolation='nearest')
plt.title("Confusion matrix")
plt.xlabel('true')
plt.show()


### print some digits ###
fig, axes = plt.subplots(15, 15, figsize=(10, 10))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(np.flipud(Xtest[i].reshape(8, 8)), cmap='binary')
    ax.text(0.05, 0.05, str(pred_test[i]), transform=ax.transAxes, 
            color='green' if (ytest[i] == pred_test[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])