import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

################################################################################################################

### plotting the death distribution ###
x = np.arange(5)
death = [109, 65, 22, 3, 1]

counts = []   #vector with the explicit count of deaths
for n in range(5) :
    for i in range(death[n]) :
        counts.append(x[n])


### Poisson function ###
mu = np.average(x, weights=death)   #sample mean as the mu of the distribution
distP = poisson(mu)


### plot ###
bins = np.linspace(-0.5, 4.5, 6)

plt.figure()
plt.hist(counts, bins=bins, density=True, histtype="step", lw=2, color="royalblue")
plt.plot(x, distP.pmf(x), color="red")
plt.title("Prussian cavalryman horse-kick deaths")
plt.xticks(np.arange(5))
plt.xlabel("deaths")
plt.ylabel("fraction of groups")
plt.show()
