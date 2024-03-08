import numpy as np

################################################################################################################

games = 10000
N = 3   #number of doors
M = 1   #doors opened by the presenter

tests = False

print("\nGames: ", games)
print("\nNumber of doors: ", N)
print("Opened by host: ", M)

#%%
### INITIALIZE THE GAME ########################################################################################

### position the car ###
doors = np.zeros(N)
car = np.random.choice(np.arange(0,N,1))   #index of the car position

doors[car] = 1

### test ###
if tests:
    print("Initial condition:")
    print(car)
    print(doors)

#%%
### CONSERVATIVE ###############################################################################################

i = 0
win = 0

for i in range(games) :
    
    ### first and only choice ###
    ch = np.random.choice(np.arange(0,N,1))   #index of the chosen door
    
    if doors[ch] == 1 :
        win += 1
        
    i += 1

prob = win / games

bayes = 1/N

### test ###
print("\nCONSERVATIVE")
if tests:
    print("Games: ", i)
    print("Victories: ", win)
print("Prob:" , round(prob*100, 3), "%")
print("Check: ", round(bayes*100, 3), "%")

#%%
### SWITCHER ###################################################################################################

i = 0
win = 0

for i in range(games) :
    
    closed = []   #contains the closed doors exept for the first choice
    
    ### first choice ###
    ch1 = np.random.choice(np.arange(0,N,1))   #index of the first choice
    
    ### presenter turn ###
    if car != ch1 :
        closed.append(car)   #the winning door remains always closed

    while len(closed) < (N-M-1) :   #the -1 is for the exclusion of the first choice
        var = np.random.choice(np.arange(0,N,1))   #indexes of the other closed doors
        if var != ch1 :
            if var not in closed:
                closed.append(var)
    
    ### second choice ###
    ch2 = np.random.choice(closed)   #index of the second choice
    
    if doors[ch2] == 1 :
        win += 1
        
    i += 1

prob = win / games

bayes = (N-1)/N * 1/(N-M-1)

### test ###
print("\nSWITCHER")
if tests:
    print("Games: ", i)
    print("Victories: ", win)
print("Prob:" , round(prob*100, 3), "%")
print("Check: ", round(bayes*100, 3), "%")

#%%
### NEWCOMER ###################################################################################################

i = 0
win = 0

for i in range(games) :
    
    closed = [car]   #contains the closed doors, in which there's the winning one
    
    ### host turn ###
    while len(closed) < (N-M) :
        var = np.random.choice(np.arange(0,N,1))   #indexes of the closed doors, choosen randomly by the host
        if var not in closed :
            closed.append(var)
    
    ### player choice ###
    ch = np.random.choice(closed)   #index of the chosen door
    
    if doors[ch] == 1 :
        win += 1
        
    i += 1

prob = win / games

bayes = 1/(N-M)

### test ###
print("\nNEWCOMER")
if tests:
    print("Games: ", i)
    print("Victories: ", win)
print("Prob:" , round(prob*100, 3), "%")
print("Check: ", round(bayes*100, 3), "%")