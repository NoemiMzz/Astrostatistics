import numpy as np
import matplotlib.pyplot as plt

################################################################################################################

N = 10000

### FUNCTIONS ##################################################################################################

def pdf(y) :
    return (10**y) * np.log(10) * 1/(10-0.1)

################################################################################################################

x = []

for i in range(N) :   #generating x data
    x.append(np.random.uniform(0.1, 10))
    
y = np.log10(x)

plt.hist(x, density=True, color="deepskyblue")   #x distribution
plt.hist(y, density=True, color="orange")   #y distribution

plt.plot(np.log10(np.linspace(0.1, 10, 100)), pdf(np.log10(np.linspace(0.1, 10, 100))), color="red")   #y pdf

#mean computation
print("Logaritm of mean of x:")
print(np.log10(np.mean(x)))
print("Mean of y:")
print(np.mean(y))

#median computation
print("\nLogaritm of median of x:")
print(np.log10(np.median(x)))
print("Median of y:")
print(np.median(y))