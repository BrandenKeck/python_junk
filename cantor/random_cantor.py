import numpy as np
import matplotlib.pyplot as plt

def random_cantor(res):
    u = 0
    for i in np.arange(1,res):
        u = u + (1/(3**i))*2*(np.random.binomial(1, 0.5, 1))
        
    return u[0]

X = []
iterations = 10000
for i in np.arange(iterations):
    X.append(random_cantor(1000))
    
plt.hist(X, bins=100)
    