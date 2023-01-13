from scipy.stats import rv_discrete
from scipy.special import factorial
from numpy import exp, sqrt

class gpoisson_gen(rv_discrete):
    def _pmf(self, k, mean, var):
        a = (sqrt(var/mean) - 1)/mean
        p = (mean / (1+a*mean))**k*((1+a*k)**(k-1)/factorial(k))*exp((-mean*(1+a*k))/(1+a*mean))
        return p

gpoisson = gpoisson_gen(name="poisson", longname='A Poisson')