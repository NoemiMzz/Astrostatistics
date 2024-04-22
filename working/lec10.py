import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

################################################################################################################

N = 100000   #number of days
p_cc = 0.5   #probability cloudy given cloudy
p_cs = 0.1   #probability cloudy given sunny
p_ss = 0.9   #probability sunny given sunny
p_sc = 0.5   #probability sunny given cloudy and viceversa

################################################################################################################

### sampling the weather ###
w = [1]   #weather, starting from a sunny day

for i in tqdm(range(1, N)):
    if w[i-1] == 0 :
        w.append(np.random.choice([0, 1], p=[p_cc, p_sc]))   #weather proposal given cloudy
    elif w[i-1] == 1 :
        w.append(np.random.choice([0, 1], p=[p_cs, p_ss]))   #weather proposal given sunny

w = np.array(w)
sunny_fraction = np.cumsum(w) / (np.arange(N) + 1)   #fraction of sunny days

plt.figure()
plt.plot(np.arange(N), sunny_fraction, color='crimson')
plt.title("Trace plot")
plt.xlabel("days")
plt.ylabel("prob")
plt.show()

plt.figure()
plt.hist(sunny_fraction, bins=int(np.sqrt(N)), density=True, color='gold')
plt.title("Sunny days")
plt.xlabel("prob")
plt.show()

#%%
### sunny days ###
burn = 25000   #burn in the first samples
sunny_burned = sunny_fraction[burn:]

# statistic with median
med = np.median(sunny_burned)
p1 = np.percentile(sunny_burned, 16)
p2 = np.percentile(sunny_burned, 84)
print("SUNNY:")
print("median = ", np.round(med, 2))

#statistic with mean
mean = np.mean(sunny_burned)
sigma = np.std(sunny_burned)
print("mean = ", np.round(mean, 2))

plt.figure()
plt.hist(sunny_burned, bins=int(np.sqrt(N)), density=True, color='gold')
plt.axvline(med, color='orangered', label="median")
plt.axvline(p1, color='orangered', ls='dotted', label="percentiles 68%")
plt.axvline(p2, color='orangered', ls='dotted')
plt.title("Sunny days")
plt.xlabel("prob")
plt.legend()
plt.show()

plt.figure()
plt.hist(sunny_burned, bins=int(np.sqrt(N)), density=True, color='gold')
plt.axvline(mean, color='orangered', label="mean")
plt.axvline(mean-sigma, color='orangered', ls='dotted', label="$\pm \sigma$")
plt.axvline(mean+sigma, color='orangered', ls='dotted')
plt.title("Sunny days")
plt.xlabel("prob")
plt.legend()
plt.show()

#%%
### cloudy days ###
cloudy_burned = 1 - sunny_fraction[burn:]

# statistic with median
med = np.median(cloudy_burned)
p1 = np.percentile(cloudy_burned, 16)
p2 = np.percentile(cloudy_burned, 84)
print("\nCLOUDY:")
print("median = ", np.round(med, 2))

#statistic with mean
mean = np.mean(cloudy_burned)
sigma = np.std(cloudy_burned)
print("mean = ", np.round(mean, 2))

plt.figure()
plt.hist(cloudy_burned, bins=int(np.sqrt(N)), density=True, color='lightskyblue')
plt.axvline(med, color='b', label="median")
plt.axvline(p1, color='b', ls='dotted', label="percentiles 68%")
plt.axvline(p2, color='b', ls='dotted')
plt.title("Cloudy days")
plt.xlabel("prob")
plt.legend()
plt.show()

plt.figure()
plt.hist(cloudy_burned, bins=int(np.sqrt(N)), density=True, color='lightskyblue')
plt.axvline(mean, color='b', label="mean")
plt.axvline(mean-sigma, color='b', ls='dotted', label="$\pm \sigma$")
plt.axvline(mean+sigma, color='b', ls='dotted')
plt.title("Cloudy days")
plt.xlabel("prob")
plt.legend()
plt.show()