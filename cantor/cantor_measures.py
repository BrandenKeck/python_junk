import copy
import numpy as np

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

iterations = 100000
for i in np.arange(iterations):
    tot = 0
    u = np.random.rand()
    cantor = frac_convert (u, 3, 50)
    if 1 not in cantor:
        tot = tot + 1
        
print(tot)