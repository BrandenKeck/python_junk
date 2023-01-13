# Import model
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.discrete.count_model as cm
from statsmodels.distributions.discrete import zinegbin

# Simulate random positive statistics
def get_x():
    x1 = np.random.randint(1, 10)
    x2 = np.random.randint(1, 20)
    x3 = np.random.randint(1, 100) / 10
    x4 = np.random.rand()
    x5 = np.random.rand()
    return np.array([x1, x2, x3, x4, x5])

def convert_params(self, mu, alpha, p):
    size = 1. / alpha * mu**(2-p)
    prob = size / (size + mu)
    return (size, prob)

# Simulate model data
def zinb_simulator(mu_params, alpha_params, p_params, w_params, N):

    X = []
    Y = []
    for i in np.arange(N):
        x = get_x()
        mu = x @ mu_params
        alpha = x @ alpha_params
        p = x @ p_params
        w = np.exp(x @ w_params) / (1 + np.exp(x @ w_params))
        y = zinegbin(mu, alpha, p, w).rvs(1)
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

# Set up ZIP data from params
N = 100000
nparams = 5
mu_params = np.array([0.011, 0.015, 0.013, 0.008, 0.005])
alpha_params = np.array([0.055, 0.111, 0.215, 0.316, 0.151])
p_params = np.array([0.222, 0.322, 0.244, 0.252, 0.226])
w_params = np.array([0.0010, 0.0020, 0.0033, 0.0043, 0.0012])
X, Y = zinb_simulator(
    mu_params,
    alpha_params,
    p_params,
    w_params,
    N
)

# Fitted model
out = cm.ZeroInflatedNegativeBinomialP(Y, X, exog_infl=X)
res = out.fit()

while True:

    # Extract real mu and w as well as simulated mu and w
    xx = get_x()
    mu_real = xx @ mu_realparams
    w_real = np.exp(xx @ w_realparams) / (1 + np.exp(xx @ w_realparams))
    mu_sim = res.predict(xx, exog_infl=xx, which='mean-main')[0]
    w_sim = (1 - res.predict(xx, exog_infl=xx, which='prob-main'))

    # Params Listed Out
    print(f'Real Rate: {mu_real}')
    print(f'Real Probs: {w_real}')
    print(f'Sim Rate: {mu_sim}')
    print(f'Sim Probs: {w_sim}')

    # ZIP distributions
    zip_real = zipoisson(mu_real, w_real)
    zip_sim = zipoisson(mu_sim, w_sim)

    # Get data for plot
    yplot_real = [zip_real.pmf(xx) for xx in np.arange(0, max(Y))]
    yplot_sim = [zip_sim.pmf(xx) for xx in np.arange(0, max(Y))]

    # Plot both
    width = 0.35
    plt.bar(np.arange(0, max(Y)), yplot_real, width)
    plt.bar(np.arange(0, max(Y))+width, yplot_sim, width)
    plt.show()

    input("Press any key to continue...")


# Yp = res.predict(X, exog_infl=X, which="mean-main")
# plt.hist([Y, Yp], log=True, bins=max(Y))
# plt.show()

# # Test Graph
# y = []
# for xx in x: y.append(zip.cdf(xx))
# plt.bar(x, y)
# plt.show()

# # Test Results
# print(f'Mean: {zip.mean()}')
# print(f'Stdev: {zip.std()}')
# print(f'rvs: {zip.rvs(1000)}')
