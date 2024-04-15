import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

################################################################################################################

N = 10000   #number of days
p_ss = 0.9   #probability sunny given sunny
p_sc = 0.5   #probability sunny given cloudy

################################################################################################################

x = np.zeros(N)

sunny_fraction = np.zeros(N)

for i in tqdm(range(N)):
    prop = np.random.uniform()   #draw a random number between 0 and 1
    
    if x[i-1] == 1:
        if prop > p_ss:
            sunny_fraction[i] = sunny_fraction[i-1] + 1
        else:
            sunny_fraction[i] = sunny_fraction[i-1]
            
    if x[i-1] == 0:
        if prop > p_sc:
            sunny_fraction[i] = sunny_fraction[i-1] + 1
        else:
            sunny_fraction[i] = sunny_fraction[i-1]

################################################################################################################

w = 0   #starting from a cloudy day
posterior = [w]   #current weather

sunny_days = np.zeros(N)   #amount of sunny days so far
sunny_fraction = np.zeros(N)   #cumulative fraction of sunny days

for i in tqdm(range(N)):
    w_prop = np.random.choice([0, 1])   #weather proposal   0 nad 1 need to be choosen with a probability of 0.5 and 0.9
    acc = np.random.uniform()   #proposal probability
    
    
    
################################################################################################################
    
    if w_prop == 1:
        if p_prop > p_ss:
            sunny_days[i] = sunny_days[i-1] + 1
            sunny_fraction[i] = sunny_days[i] / i
        else:
            sunny_days[i] = sunny_days[i-1]
            sunny_fraction[i] = sunny_days[i] / i
            
    if w_prop == 0:
        if p_prop > p_sc:
            sunny_days[i] = sunny_days[i-1] + 1
            sunny_fraction[i] = sunny_days[i] / i
        else:
            sunny_days[i] = sunny_days[i-1]
            sunny_fraction[i] = sunny_days[i] / i

