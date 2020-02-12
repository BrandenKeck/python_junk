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
    
    if np.abs(c-c0) < np.abs(c-c1):
        return np.abs(c-c0)
    else:
        return np.abs(c-c1)
    
X = np.linspace(0,1,100000)
Y = []
for x in X:
    xx = frac_convert(x, 3, 100)
    Y.append(nearest_cantor(xx, 100))
    
plt.plot(X, Y, 'k-')