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

################################################################################################################

plot_by_me = True

#%%
### UNSUPERVISED ###############################################################################################

digits = load_digits()   #load the data
print("\nImport data:")
print(digits.images.shape)
print(digits.keys())
print("\n")

data, target = digits.data, digits.target


### isomap for dim reduction ### 
embedding = Isomap(n_components=2)
digits_transformed = embedding.fit_transform(data)
print("\nReduced dimensions:")
print(digits_transformed.shape)
print("\n")

if plot_by_me:
    plt.figure()
    plt.scatter(digits_transformed[:,0], digits_transformed[:,1], c=target, alpha=0.5, cmap='tab10')
    plt.colorbar(label='digit label', ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title("ISOMAP")
    plt.show()
else:
    plot_embedding(digits_transformed, "ISOMAP")
    
#%%
### SUPERVISED #################################################################################################

#split the data in train and validation
Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2)

#training the data
clf = LogisticRegression(solver='sag').fit(Xtrain, ytrain)


### prediction ###
pred_train = clf.predict(Xtrain)   #train set
pred_test = clf.predict(Xtest)   #test set

#visualize errors in the validation set
for i in range(len(Xtest)):
    if pred_test[i] == ytest[i]:
        print(pred_test[i], ytest[i])
    else:
        print(pred_test[i], ytest[i], "  <--")
print("\n")        

print("Accuracy of training digits:")   #computing the accuracy score
print(accuracy_score(ytrain, pred_train))   #train set
print("\n")
print("Accuracy of test digits:")
print(accuracy_score(ytest, pred_test))   #test set
print("\n")

print("Confusion matrix of training digits:")   #computing the confusion matrix
print(confusion_matrix(ytrain, pred_train))   #train set
print("\n")
print("Confusion matrix of test digits:")
print(confusion_matrix(ytest, pred_test))   #test set
print("\n")

#visualize test confusion matrix
plt.imshow(np.log(confusion_matrix(ytest, pred_test)), cmap='BuGn', interpolation='nearest');
plt.ylabel('true')
plt.xlabel('predicted');


### print some digits ###
fig, axes = plt.subplots(15, 15, figsize=(10, 10))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(np.flipud(Xtest[i].reshape(8, 8)), cmap='binary')
    ax.text(0.05, 0.05, str(pred_test[i]), transform=ax.transAxes, 
            color='green' if (ytest[i] == pred_test[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])