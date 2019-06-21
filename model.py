from sklearn.base import BaseEstimator, ClassifierMixin
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score
import numpy as np


class auto_encoder_model(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 threshold=3e10,
                 refit=False,
                 Xshape = 12):
        self.threshold = threshold
        self.Xshape = Xshape
        autoencoder = Sequential()
        autoencoder.add(Dense(10, input_shape=(self.Xshape,), activation='relu'))
        autoencoder.add(Dense(10, activation='relu'))
        autoencoder.add(Dense(5, activation='relu'))
        autoencoder.add(Dense(10, activation='relu'))
        autoencoder.add(Dense(self.Xshape, activation='sigmoid'))
        autoencoder.compile(loss='mse',
                            optimizer=keras.optimizers.Adam())
        autoencoder.load_weights('autoencoder.h5')
        self.autoencoder = autoencoder
        self.refit = refit

    def fit(self, X, y=None):
        if self.refit:
            print('Refitting model')
            self.autoencoder.fit(x=X,
                                 y=X,
                                 batch_size=5,
                                 epochs=5,
                                 verbose=0)

        return self

    def predict(self, X):
        pred_autoencoder = self.autoencoder.predict(X)
        sqerr = (pred_autoencoder - X) ** 2
        mse = np.mean(sqerr, axis=1)
        predict = np.where(mse > self.threshold, 1, 0)
        return predict

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = y
        f1 = f1_score(y_true, y_pred)
        return f1