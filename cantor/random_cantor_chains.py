import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import copy

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
        return np.sign(c-c0)*np.abs(c-c0)
    else:
        return np.sign(c-c1)*np.abs(c-c1)


def random_cantor(res):
    u = 0
    for i in np.arange(1,res):
        u = u + (1/(3**i))*2*(np.random.binomial(1, 0.5, 1))
        
    return u


def animate(i, X, Y1, Y2, Y3):
    if i <= 90:
        x = X[:i]
        y1 = Y1[:i]
        y2 = Y2[:i]
        y3 = Y3[:i]
    else:
        x = X[0:90]
        y1 = Y1[i-90:i]
        y2 = Y2[i-90:i]
        y3 = Y3[i-90:i]
    
    line1.set_data(x, y1)
    line2.set_data(x, y2)
    line3.set_data(x, y3)


fig = plt.figure()
ax = plt.axes(xlim=(0,0.2), ylim=(-50,50))
line1, = ax.plot([], [], 'r.')
line2, = ax.plot([], [], 'b.')
line3, = ax.plot([], [], 'g.')

X = np.linspace(0, 10, 10000)
Y1 = np.zeros(len(X))
Y2 = np.zeros(len(X))
Y3 = np.zeros(len(X))
for i in np.arange(1, len(X)):
    Y1[i] = Y1[i-1] + (2*np.random.binomial(1, 0.5, 1) - 1)*np.random.beta(0.5,0.5)
    Y2[i] = Y2[i-1] + random_cantor(100) - 0.5
    
    cant = frac_convert (Y1[i], 3, 100)
    Y3[i] = Y3[i-1] + nearest_cantor(cant, 100)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, fargs=(X, Y1, Y2, Y3), frames=10000, interval=1)

#plt.plot(X,Y,'g.')

plt.show()