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
    x.append(np.random.uniform(0.1, 10))   #uniformly distributed
    
y = np.log10(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4))
ax1.hist(x, density=True, color="deepskyblue", alpha=0.7)   #x distribution
ax1.set_xlabel("x")
ax1.set_title("Uniform distribution")
ax2.hist(y, density=True, color="orange", alpha=0.7)   #y distribution
ax2.set_xlabel("y")
ax2.set_title("Transformed distribution")

plt.plot(np.log10(np.linspace(0.1, 10, 100)), pdf(np.log10(np.linspace(0.1, 10, 100))), color='crimson')  #y pdf

#mean computation
print("Logaritm of mean of x:", round(np.log10(np.mean(x)), 3))
print("Mean of y:", round(np.mean(y), 3))

#median computation
print("\nLogaritm of median of x:", round(np.log10(np.median(x)), 3))
print("Median of y:", round(np.median(y), 3))