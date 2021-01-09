import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from matplotlib import animation

def make_signal1(X):
    Y1 = np.sin(2 * np.pi * X) - np.cos(np.pi*X)*np.sin(4*np.pi*X)
    Y2 = np.zeros(len(X))
    for i in np.arange(len(X)):
        Y2[i] = np.sin(2 * np.pi * X[i]) - np.cos(np.pi*X[i])*np.sin(4*np.pi*X[i]) + np.random.normal(0, 0.5)
    
    return Y1, Y2

def make_signal2(X):
    dy = 0.001
    Y1 = np.zeros(len(X))
    Y2 = np.zeros(len(X))
    for i in np.arange(1, len(X)):
        if (abs(Y1[i-1]) >= 1.5):
            dy = -dy
        Y1[i] = Y1[i-1] + dy
        Y2[i] = Y1[i] + np.random.normal(0, 2)
    
    return Y1, Y2

def make_signal3(X):
    amplitude = 1
    interval = 1000
    counter = 0
    Y1 = np.zeros(len(X))
    Y2 = np.zeros(len(X))
    for i in np.arange(1, len(X)):
        if (counter == interval):
            counter = 0
            amplitude = -amplitude
        Y1[i] = amplitude
        Y2[i] = amplitude + np.random.normal(0, 2)

        counter = counter + 1     
    
    return Y1, Y2
    
def average_30pt(X):
    Y = np.zeros(len(X))
    for i in np.arange(30, len(X)):
        Y[i] = np.average(X[i-30:i])
        
    return Y

def low_pass(X, alpha):
    Y = np.zeros(len(X))
    Y[0] = X[0]
    for i in np.arange(1, len(X)):
        Y[i] = alpha*X[i] + (1-alpha)*Y[i-1]
        
    return Y
    
def recursive_lp(X, alpha, n):
    Y = np.zeros(len(X))
    Y[0] = X[0]
    for i in np.arange(1, len(X)):
        Y[i] = alpha*X[i] + (1-alpha)*Y[i-1]
    
    if n == 1:
        return Y
    else:
        Y = recursive_lp(Y, alpha, n-1)
    
    return Y

def recursive_fivepass_lp(X, alpha, n):
    Y = np.zeros(len(X))
    Y[0:5] = X[0:5]
    for i in np.arange(5, len(X)):
        Y[i] = alpha*X[i] + ((1-alpha)/5)*(Y[i-1] + Y[i-2] + Y[i-3] + Y[i-4] + Y[i-5])
    
    if n == 1:
        return Y
    else:
        Y = recursive_fivepass_lp(Y, alpha, n-1)
    
    return Y
    

# animation function.  This is called sequentially
def animate(i, X, Y1, Y2, Y3, Y4, Y5):
    if i <= 1900:
        x = X[:i]
        y1 = Y1[:i]
        y2 = Y2[:i]
        y3 = Y3[:i]
        y4 = Y4[:i]
        y5 = Y5[:i]
    else:
        x = X[0:1900]
        y1 = Y1[i-1900:i]
        y2 = Y2[i-1900:i]
        y3 = Y3[i-1900:i]
        y4 = Y4[i-1900:i]
        y5 = Y5[i-1900:i]
    
    line1.set_data(x, y2)
    line2.set_data(x, y1)
    line3.set_data(x, y3)
    line4.set_data(x, y4)
    line5.set_data(x, y5)
    return line1, line2, line3, line4, line5

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line1, = ax.plot([], [], 'r-')
line2, = ax.plot([], [], 'k--')
line3, = ax.plot([], [], 'g-')
line4, = ax.plot([], [], 'b-')
line5, = ax.plot([], [], 'm-')

X = np.linspace(0, 10, 10000)
[Y1, Y2] = make_signal1(X)
Y3 = low_pass(average_30pt(Y2), 0.1)
Y4 = recursive_lp(average_30pt(Y2), 0.1, 20)
#Y5 = low_gain(Y2, 0.1)
Y5 = recursive_fivepass_lp(Y2, 0.1, 5)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, fargs=(X, Y1, Y2, Y3, Y4, Y5), frames=20000, interval=1, blit=True)

plt.show()

sqr_err = np.sum((Y1-Y3)**2)
print(sqr_err)