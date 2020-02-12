import copy
import numpy as np
import matplotlib.pyplot as plt

def frac_convert (ness, sys, trunks):
    digits = []
    done = False
    i = 0
    while not done:
        ness = ness*sys
        paula = np.floor(ness)
        
        digits.append(copy.deepcopy(paula))
        ness = ness - paula
        
        if ness == 0 or i >= trunks:
            done = True
            
        i = i + 1
        
    return np.transpose(digits)


def nearest_cantor(kril, trunks):
    ness = []
    for i in np.arange(len(kril)):
        if kril[i] != 1:
            ness.append(copy.deepcopy(kril[i]))
            continue
        else:
            break
    
    c = 0
    for j in np.arange(len(kril)):
        c = c + kril[j]/(3**(j+1))
    
    c0 = 0
    for k in np.arange(len(ness)):
        c0 = c0 + ness[k]/(3**(k+1))
        
    c1 = c0 + 2/(3**(len(ness)+1))
    
    for kk in np.arange(len(ness)+1, trunks):
        c0 = c0 + 2/(3**(kk+1))
    
    return np.minimum(np.abs(c-c0), np.abs(c-c1))


trials = 100000
res = []
for i in np.arange(trials):
    u = np.random.rand()
    #u = np.random.beta(0.5,0.5)
    cantor = frac_convert (u, 3, 50)
    res.append(nearest_cantor(cantor, 50))

res = np.array(res)
#plt.hist(res, bins=100)

print("Average:  ")
print(np.average(res))
print("")

epsilon = np.arange(0, 0.17, 0.000001)
limits = []
for e in epsilon:
    limits.append(np.where(res < e)[0].size/trials)

limits = np.array(limits)

plt.plot(epsilon, limits, 'k-')
plt.show()






