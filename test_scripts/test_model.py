# Import model
import warnings
import numpy as np
import keras_tuner
import tensorflow as tf
from tensorflow import keras
from more_itertools import locate
from keras.models import load_model
from tensorflow_addons.metrics import RSquare
from scipy.stats import uniform, loguniform, randint
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.metrics import AUC, RootMeanSquaredError
from sklearn.feature_selection import RFECV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet
)
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

# Object definition
class test_model():

    def __init__(self, start, prefactors, postfactors, response, data):

        # Init distribution parameters
        self.model = None
        self.start = start
        self.prefactors = prefactors
        self.postfactors = postfactors
        self.response = response
        self.data = data
        self.structure = None
        self.eval = None
        self.params = None
        self.selector = None

    ###
    # DATA MANIPULATION
    ###

    def set_numerical_data(self):

        # Set Responses
        idx = self.data.index
        Y = self.data.loc[idx[self.start+1]:, self.response].to_numpy()
        # Y = 1*np.greater(self.data.loc[idx[self.start+1]:, self.response].to_numpy(), 0)

        # Set-up Data
        X = self.data.loc[idx[self.start]:idx[len(idx)-2], self.prefactors].to_numpy()

        # Establish Dataset
        self.X = X
        self.Y = Y

    def set_categorical_data(self):
        X = self.X
        idx = self.data.index
        for factor in self.postfactors:
            catcol = self.data.loc[idx[self.start+1]:, factor[0]].to_numpy().astype(int)
            to_onehot = np.zeros((catcol.size, factor[1]))
            to_onehot[np.arange(catcol.size), catcol] = 1
            X = np.concatenate((X, to_onehot), axis=1)
        self.X = np.concatenate((self.X, X), axis=1)

    ###
    # RANDOM FOREST PIPELINE
    ###

    def feature_selection(self):
        selectors = []
        self.set_numerical_data()
        selector = RFECV(RandomForestRegressor(), step=1, cv=5)
        selector.fit(self.X, self.Y)
        self.selector = selector.support_

    def select_data(self):
        self.set_numerical_data()
        self.X = self.X[:, self.selector]
        self.set_categorical_data()

    def build_randforest(self):
        trainX, testX, trainY, testY = train_test_split(self.X, self.Y, test_size=0.2, shuffle= True)
        mod = RandomForestRegressor(n_estimators=1000)
        mod.fit(trainX, trainY)
        predY = mod.predict(testX)
        print(f'Mean Abs Error: {mean_absolute_error(testY, predY)}')
        return mod

    ###
    # NEURAL NETWORKS PIPELINE
    ###

    def build_model(self, hp):
        # dropout=0.0, reg=1e-3, lr=1e-4, delta=1.0):
        model = tf.keras.Sequential([keras.Input(shape=(self.X.shape[1],))])
        for lay in self.structure:
            model.add(keras.layers.Dense(
                units=lay,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(
                    hp.Float("reg", min_value=1e-6, default=1e-4, max_value=1e-2, sampling="log")
                )
            ))
            #model.add(keras.layers.Dense(units=lay, activation="relu"))
            model.add(
                keras.layers.Dropout(
                    hp.Float("dropout", min_value=0.0, default=0.2, max_value=0.5, sampling="linear")
                )
            )
        model.add(keras.layers.Dense(units=1, activation="linear"))
        optim = keras.optimizers.Adam(
            learning_rate=hp.Float("lr", min_value=1e-6, default=1e-4, max_value=1, sampling="log")
        )
        loss_function = tf.losses.Huber(
            delta=hp.Float("delta", min_value=0.5, default=1.0, max_value=5.5, sampling="linear")
        )
        model.compile(optimizer=optim,
              loss=loss_function,
              metrics=['accuracy',
                        'mean_squared_error',
                        'mean_absolute_error',
                        RootMeanSquaredError(),
                        RSquare(), AUC()])
        return model

    def paramsearch(self, epochs=2000, best_structure=[4096, 4096]):

        # Init params
        warnings.filterwarnings("ignore")
        es = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=1000)
        mc = tf.keras.callbacks.ModelCheckpoint('structure_max.h5', monitor='mean_squared_error', save_best_only=True)

        # Layer Analysis - One Shot no Crossvalidation
        if not best_structure:
            print("-- Begin Layer Search --")
            layers = self.generate_layers()
            best_loss = 1e99
            best_structure = layers[0]
            dataset = tf.data.Dataset.from_tensor_slices((self.X.astype('float32'), self.Y.astype('float32'))).shuffle(len(self.Y)).batch(len(self.Y))
            for l in layers:
                print(f'Training on {l}')
                self.structure = l
                mod = self.build_model(keras_tuner.HyperParameters())
                mod.fit(
                    dataset,
                    verbose=0,
                    epochs=epochs,
                    callbacks=[es, mc],
                    steps_per_epoch=None,
                    use_multiprocessing = True,
                    batch_size=np.min([self.X.shape[0], 32])
                )
                mod = load_model('structure_max.h5')
                eval = mod.evaluate(self.X, self.Y, verbose=0)
                if eval[0] < best_loss:
                    best_loss = eval[0]
                    best_structure = l
                print(f'Training Complete | Best MSE: {best_loss}, Best Struct: {best_structure}')

        # Tune parameters for the specific structure - CV confirmation
        print("-- Begin Hyperparameter Search --")
        # mod = KerasClassifier(build_fn=self.build_model,
        #     layers=best_structure,
        #     batch_size=np.min([128, np.min([self.X.shape[0], 256])]),
        #     epochs=epochs,
        #     verbose=0)
        # params={
        #     'dropout': uniform(loc=0.2, scale=0.3),
        #     'reg': loguniform(a=1e-6, b=1e-3),
        #     'lr': loguniform(a=1e-5, b=1e-1),
        #     'delta': uniform(loc=0.25, scale=3)
        # }
        # grid = RandomizedSearchCV(estimator=mod,
        #     param_distributions=params,
        #     n_iter=n_iter,
        #     verbose=2,
        #     cv=3)
        # result = grid.fit(self.X, self.Y)
        self.structure = best_structure
        tuner = keras_tuner.RandomSearch(
            self.build_model,
            objective="loss",
            max_trials = 10,
            directory="my_dir",
            project_name="paramsearch",
        )
        # tuner = keras_tuner.Hyperband(
        #             hypermodel = self.build_model,
        #              objective = "loss",
        #              hyperband_iterations=1,
        #              factor=10,
        #              max_epochs = epochs,
		# 			 project_name='hyperbands')
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=self.build_model,
            objective="loss",
            max_trials=10,
            project_name="bayesian",
        )
        tuner.search(self.X, self.Y,
                    callbacks=[es],
                    epochs=epochs,
                    batch_size=np.min([self.X.shape[0], 32]),
                    use_multiprocessing = True,
                    verbose=0)
        self.params = tuner.get_best_hyperparameters()[0].values

        # Return Parameters
        # self.params = result.best_params_
        # self.params['layers'] = best_structure
        # print(f"Parameters Locked: {self.params}")
        return self.params

    # def set_model(self):
    #     if self.params:
    #         self.model = self.build_model(
    #             layers=self.params["layers"],
    #             dropout=self.params["dropout"],
    #             reg=self.params["reg"],
    #             lr=self.params["lr"],
    #             delta=self.params["delta"]
    #         )
    #     else: self.model = self.build_model()

    def train(self, hps):
        warnings.filterwarnings("ignore")
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000)
        mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True)
        dataset = tf.data.Dataset.from_tensor_slices((self.X.astype('float32'), self.Y.astype('float32'))).shuffle(len(self.Y)).batch(len(self.Y))
        self.model = self.build_model(hps)
        self.model.fit(
            dataset,
            verbose=0,
            epochs=10000,
            callbacks=[es, mc],
            steps_per_epoch=None,
            use_multiprocessing = True,
            batch_size=np.min([self.X.shape[0], 32])
        )
        self.model = load_model('best_model.h5')
        self.eval = self.model.evaluate(self.X, self.Y, verbose=0)

    def cross_validate(self):
        cvscores = []
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=12345)
        for train, test in kfold.split(self.X, self.Y):
        	self.model.fit(self.X[train], self.Y[train], epochs=25, batch_size=10, verbose=0)
        	scores = self.model.evaluate(self.X[test], self.Y[test], verbose=0)
        	cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    def predict(self, idx):
        t = tf.constant(np.array(X[idx, :]).reshape(1, np.array(X[idx, :]).shape[0]))
        dataset = tf.data.Dataset.from_tensors(t)
        response = self.model.predict(dataset, verbose=False)
        print(X[idx, :])
        print(response)

    def generate_layers(self, nlaymin=2, nlaymax=3, minpow=8, maxpow=12):
        layers = []
        for i in np.arange(2**minpow, 2**maxpow):
            binary = [*"{0:b}".format(i)]
            binary = [str(x) for x in np.zeros(maxpow-len(binary), dtype=int)] + binary
            if binary.count('1') >= nlaymin and binary.count('1') <= nlaymax:
                ll = [2**(maxpow-xx) for xx in list(locate(binary, lambda x: x == '1'))]
                if all([l > 2**minpow for l in ll]):
                    layers.append(ll)
            if binary.count('1') == 1:
                ll = 2**(maxpow-binary.index('1'))
                if ll > 2**minpow:
                    for ii in np.arange(nlaymin, nlaymax+1):
                        if ii == 1: continue
                        layers.append(ii * [ll])
        return sorted(layers, key=len)
