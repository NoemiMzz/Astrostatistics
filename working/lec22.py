import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

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
### FUNCTIONS ##################################################################################################

def train_test_model(hparams):
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(4, )),   #input layer
        keras.layers.Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACTIVATION], name="hidden"),   #hidden layer
        keras.layers.Dense(1, activation='sigmoid', name="output") ])   #output layer
    
    if hparams[HP_OPTIMIZER] == 'adam':
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hparams[HP_L_RATE]),
                           loss='binary_crossentropy',   #loss function for a binary classifier
                           metrics = ['accuracy'])
    elif hparams[HP_OPTIMIZER] == 'SGD':
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=hparams[HP_L_RATE]),
                           loss='binary_crossentropy',   #loss function for a binary classifier
                           metrics = ['accuracy'])
    
    model.fit(Xtrain, ytrain, epochs=5)
    _, accuracy = model.evaluate(Xtest, ytest)
    
    yprob = model.predict(Xtest)
    ypred = np.where(yprob<=0.5, 0, 1)
    
    test_accuracy.append(metrics.accuracy_score(ytest, ypred))
    
    return accuracy
    
    
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


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
### NEURAL NETWORK #############################################################################################

print("\n--- FIRST NEURAL NETWORK ---")

X = (StandardScaler().fit_transform(colors)).T  #rescale the data

Xtrain, Xtest, ytrain, ytest = train_test_split(X, source, test_size=0.30)

keras.backend.clear_session()   #reset the NN
np.random.seed(42)
tf.random.set_seed(42)

#set the structure of my NN
model = keras.Sequential([
    keras.layers.InputLayer(shape=(4, )),   #input layer
    keras.layers.Dense(5, activation='tanh', name="hidden"),   #hidden layer with 5 nodes
    keras.layers.Dense(1, activation='sigmoid', name="output") ])   #output layer

print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),   #gradient descent
                   loss='binary_crossentropy',   #loss function for a binary classifier
                   metrics = ['accuracy'])

print("\nTraining the classifier:")
model.fit(Xtrain, ytrain, epochs=10)

print("\nPredicting on test set:")
yprob = model.predict(Xtest)
ypred = np.where(yprob<=0.5, 0, 1)

fpr, tpr, thresh = roc_curve(ytest, yprob)   #compute true and false positive rate

#plot the ROC curve
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, lw=2, color='navy')
if zoom:
    plt.ylim(min_zoom, 1)  
plt.title("ROC curve - before optimization")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()

print("\nCompleteness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly


#%%
### NN OPTIMIZATION ############################################################################################

print("\n\n\n--- OPTIMIZATION ---\n")

keras.backend.clear_session()   #reset the NN
np.random.seed(42)
tf.random.set_seed(42)

#choosing what to optimize
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([5, 7, 9]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'SGD']))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu', 'tanh']))
HP_L_RATE= hp.HParam('learning_rate', hp.Discrete([10**(-2), 10**(-2.5), 10**(-3)]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
      hparams=[HP_NUM_UNITS, HP_OPTIMIZER, HP_ACTIVATION, HP_L_RATE],
      metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')] )

#perform the optimization
test_accuracy = []
hyperparams = []
session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
    for optimizer_name in HP_OPTIMIZER.domain.values:
        for activation in HP_ACTIVATION.domain.values:
            for l_rate in HP_L_RATE.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_OPTIMIZER: optimizer_name,
                    HP_ACTIVATION: activation, 
                    HP_L_RATE: l_rate, }
                hyperparams.append([num_units, optimizer_name, activation, l_rate])
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1


#%%
### OPTIMIZED NN ###############################################################################################

print("\n\n\n--- OPTIMIZED NEURAL NETWORK ---")

BEST = int(np.argmax(test_accuracy))   #find the index of the best 

#print the optimized parameters
print("\nOptimized parameters")
print("Number of nodes:", hyperparams[BEST][0])
print("Activation function:", hyperparams[BEST][2])
print("Optimizer:", hyperparams[BEST][1])
print("Learning rate:", np.format_float_scientific(hyperparams[BEST][3], precision=3))

keras.backend.clear_session()   #reset the NN
np.random.seed(42)
tf.random.set_seed(42)


### retrain the NN with optimized parameters ###
model = keras.Sequential([
    keras.layers.InputLayer(shape=(4, )),   #input layer
    keras.layers.Dense(hyperparams[BEST][0], activation=hyperparams[BEST][2], name="hidden"),   #hidden layer
    keras.layers.Dense(1, activation='sigmoid', name="output") ])   #output layer

print("\n", model.summary())

if hyperparams[BEST][1] == 'adam':
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hyperparams[BEST][3]),
                       loss='binary_crossentropy',   #loss function for a binary classifier
                       metrics = ['accuracy'])
elif hyperparams[BEST][1] == 'SGD':
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=hyperparams[BEST][3]),
                       loss='binary_crossentropy',   #loss function for a binary classifier
                       metrics = ['accuracy'])

print("\nTraining the classifier:")
model.fit(Xtrain, ytrain, epochs=10)

print("\nPredicting on test set:")
yprob = model.predict(Xtest)
ypred = np.where(yprob<=0.5, 0, 1)

fpr, tpr, thresh = roc_curve(ytest, yprob)   #compute true and false positive rate

#plot the ROC curve
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, lw=2, color='navy')
if zoom:
    plt.ylim(min_zoom, 1)  
plt.title("ROC curve - after optimization")
plt.xlabel("false positive rate")    
plt.ylabel("true positive rate") 
plt.show()

print("\nCompleteness:", np.round(metrics.recall_score(ytest, ypred), 4))   # tp / (tp + fn)
print("Contamination:", np.round(1-metrics.precision_score(ytest, ypred), 4))   # fp / (tp + fp)
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))   #fraction labeled correctly

print("\nConfusion matrix:")
print(confusion_matrix(ytest, ypred))   #computing the confusion matrix


### visualize on the whole set ###
print("\nPredicting on the whole set:")
prob_fin = model.predict(X)
pred_fin = np.where(prob_fin<=0.5, 0, 1)

predQ = np.where(pred_fin==1)[0]
predG = np.where(pred_fin==0)[0]

#plotting the histograms
fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(6,6))
axs = axs.flatten()
fig.suptitle("Classification with NN", fontsize='14', y=0.96)

for i in range(4):
    axs[i].hist(colors[i][indQ], bins=int(np.sqrt(N)), density=True, histtype='step', lw=2, color='c', label="quasars")
    axs[i].hist(colors[i][indG], bins=int(np.sqrt(N)), density=True, histtype='step', lw=2, color='orchid', label="galaxies")
    axs[i].hist(colors[i][predQ], bins=int(np.sqrt(N)), density=True, lw=2, color='c', alpha=0.5, label="predicted q")
    axs[i].hist(colors[i][predG], bins=int(np.sqrt(N)), density=True, lw=2, color='orchid', alpha=0.5, label="predicted g")
    axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[i].set_title(titles[i])
  
axs[0].set_xlim(-0.5, 2.5)
axs[3].legend();
