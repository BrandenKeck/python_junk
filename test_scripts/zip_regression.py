# Import model
import numpy as np
import statsmodels.discrete.count_model as cm

# Object definition
class zip_model():

    def __init__(self, start, prefactors, postfactors, response, data):

        # Set-up Data
        idx = data.index
        preF = data.loc[idx[start]:idx[len(idx)-2], prefactors].to_numpy()
        postF = data.loc[idx[start+1]:, postfactors].to_numpy()
        self.X = np.concatenate((preF, postF), axis=1)
        self.Y = data.loc[idx[start+1]:, response].to_numpy()

        # Init distribution parameters
        self.mu = None
        self.w = None
        self.model = None

        # Fit Model
        self.fit()

    def fit_new(self, start, factors, response, data):
        idx = data.index
        self.X = data.loc[idx[start]:, factors].to_numpy()
        self.Y = data.loc[idx[start]:, response].to_numpy()
        self.fit()

    def fit(self):
        mod = cm.ZeroInflatedPoisson(self.Y, self.X, exog_infl=self.X)
        self.model = mod.fit(maxiter=1e5, disp=0)

    def predict(self, X):
        self.mu = self.model.predict(X, exog_infl=X, which='mean-main')[0]
        self.w = (1 - self.model.predict(X, exog_infl=X, which='prob-main'))
        return({"mu": self.mu, "w": self.w})

    def get_dist(self):
        pass # TODO
