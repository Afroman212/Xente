import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
import xgboost as xgb
from model import auto_encoder_model
from preprocess import preprocess_data
import pandas as pd
sns.set()

data = preprocess_data(training='train.csv', testing='test.csv')

_, trainX, trainy, trainX_res, trainy_res = data.preprocess_train()

RFC_res = RandomForestClassifier()
RFC_res_params = dict(n_estimators=np.arange(100, 600, 100),
                      max_features=['sqrt', 'log2'],
                      class_weight=['balanced', 'balanced_subsample', None])
grid_RFC = GridSearchCV(estimator=RFC_res,
                        param_grid=RFC_res_params,
                        cv=5,
                        scoring='f1',
                        n_jobs=-1)
grid_RFC.fit(trainX_res, trainy_res)
best_RFC = grid_RFC.best_estimator_
print('Random_Forest_best', f1_score(trainy, best_RFC.predict(trainX)))
joblib.dump(best_RFC, 'RFC.pkl')
# run_test(RFC_res, 'RFCSMOTE')


xboost_res = xgb.XGBClassifier()
xgb_res_params = dict(gamma=np.arange(0, 10, 1),
                      max_depth=np.arange(2, 10, 1),
                      alpha=np.arange(1, 5, 1))
grid_xgb = GridSearchCV(estimator=xboost_res,
                        param_grid=xgb_res_params,
                        cv=5,
                        scoring='f1',
                        n_jobs=-1)
grid_xgb.fit(trainX_res, trainy_res)
best_xgb = grid_xgb.best_estimator_
print('XGB_best', f1_score(trainy, best_xgb.predict(trainX)))
joblib.dump(best_xgb, 'XGB.pkl')
# run_test(boost_res, 'boostSMOTE')

params = dict(threshold=np.arange(2.5, 3.5, 0.1)*1e10)
autoencoder = auto_encoder_model(Xshape=trainX.shape[1])
grid_autoencoder = GridSearchCV(estimator=autoencoder,
                                param_grid=params,
                                cv=5,
                                scoring='f1')
grid_autoencoder.fit(trainX_res, trainy_res)
best_autoencoder = grid_autoencoder.best_estimator_
print('Autoencoder_best', f1_score(trainy, best_autoencoder.predict(trainX)))