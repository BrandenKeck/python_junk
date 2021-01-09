import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

def counter(num, sys):
    
    num[len(num)-1] = num[len(num)-1] + 1
    stop = False
    
    a = np.arange(len(num))
    for i in a[::-1]:
        if num[0] > (sys-1):
            stop = True
            break;
        
        if num[i] > (sys-1):
            num[i] = 0
            num[i-1] = num[i-1] + 1
    
    return num, stop
    

def get_number(num, sys):
    onett = 0
    for i in np.arange(len(num)):
        onett = onett + num[i]/(sys**(i+1))
        
    return onett


def cantor_numbers(digits, sys, exclude):
    C = []
    ness = np.zeros(digits)
    C.append(copy.deepcopy(ness))
    
    stop = False
    while not stop:
        ness, stop = counter(ness, sys)
        
        paula = False
        for e in exclude:
            if e in ness:
                paula = True
        
        if not paula:
            C.append(copy.deepcopy(ness))

    cantor = []
    for c in C:
        cantor.append(get_number(c, sys))
        
    return cantor

def append_graph(C, jeff):
    Data = []
    for i in C:
        for j in C:
            #Data.append([i,j**2+j])
            #Data.append([1/(i+1),1/(j+1)])
            #Data.append([np.cos(2*np.pi*i),np.cos(2*np.pi*j)])
            #Data.append([np.cos(2*np.pi*i),np.sin(2*np.pi*j)])
            #Data.append([np.sin(2*np.pi*i),np.sin(2*np.pi*j)])
#            if i!=0:
#                Data.append([np.sqrt(i**2 + j**2), np.arctan(j/i)])
#                Data.append([-np.sqrt(i**2 + j**2), np.arctan(j/i)])
#                Data.append([np.sqrt(i**2 + j**2), -np.arctan(j/i)])
#                Data.append([-np.sqrt(i**2 + j**2), -np.arctan(j/i)])
                #Data.append([np.arctan(j/i), np.sqrt(i**2 + j**2)])
             
            #Data.append([i**(1/2),np.cos(j)])
            #Data.append([i,np.exp(j)])
            Data.append([i,j])
            
            #Data.append([i,0])
            
    Data = np.array(Data)    
    x, y = Data.T
    
    #sns.kdeplot(x, y, cmap="Reds", shade=True, shade_lowest=True,)
    #sns.plt.show() 
    
    plt.plot(x, y, jeff)

def temp_density(d,b):
    Data = np.array(d)
    x, y = Data.T
    k = kde.gaussian_kde(Data.T)
    xi, yi = np.mgrid[x.min():x.max():b*1j, y.min():y.max():b*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='inferno')

def temp_heatmap(x,y,b):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=b)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
 
if __name__ == "__main__":
    A = cantor_numbers(8, 3, [1])
    append_graph(A, 'r.')
    
#    B = cantor_numbers(6, 4, [1])
#    append_graph(B, 'b.')
    
#    C = cantor_numbers(6, 5, [1, 2, 3])
#    append_graph(C, 'g.')
    
    
    