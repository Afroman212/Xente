import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from model import auto_encoder_model
import pandas as pd

sns.set()
sm = SMOTE(sampling_strategy=0.50, random_state=42, k_neighbors=3)


def preprocess(dataset):
    df = pd.read_csv(dataset)
    proc_df = df.drop(['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId'], axis=1)
    proc_df['TransactionStartTime'] = pd.to_datetime(proc_df['TransactionStartTime'], format='%Y-%m-%dT%H:%M:%S')
    le = LabelEncoder()
    enc_df = proc_df[['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
                      'ProductCategory', 'ChannelId', 'PricingStrategy']]
    nenc_df = proc_df.drop(['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
                             'ProductCategory', 'ChannelId', 'PricingStrategy'], axis=1)
    enc_df = enc_df.apply(le.fit_transform)
    proc_df = enc_df.join(nenc_df)
    proc_df['DayOfTrans'] = proc_df['TransactionStartTime'].dt.dayofweek
    proc_df['HourOfTrans'] = proc_df['TransactionStartTime'].dt.hour
    proc_df['PriceDiff'] = proc_df['Amount'] - proc_df['Value']
    proc_df = proc_df.drop(['TransactionStartTime'], axis=1)

    try:
        Tid = df['TransactionId'].to_frame()
        X = proc_df.drop(['FraudResult'], axis=1)
        y = proc_df['FraudResult']
        X_res, y_res = sm.fit_resample(X, y)
        X_res = pd.DataFrame(data=X_res, columns=X.columns)
        return Tid, X, y, X_res, y_res
    except KeyError:
        Tid = df['TransactionId'].to_frame()
        Xt = proc_df
        return Tid, Xt, None, None, None


def run_test(model, name):
    test, X, _ = preprocess('test.csv')
    fraud = model.predict(X)
    test['FraudResult'] = fraud
    test.to_csv('test_submission_%s.csv' % name, index=False)


_, trainX, trainy, trainX_res, trainy_res = preprocess('training.csv')

RFC_res = RandomForestClassifier()
RFC_res_params = dict(n_estimators=np.arange(100, 600, 100),
                     max_features=['sqrt', 'log2'],
                      class_weight=['balanced', 'balanced_subsample', None])
grid_RFC = GridSearchCV(estimator=RFC_res,
                        param_grid=RFC_res_params,
                        cv=5,
                        scoring='f1',
                        n_jobs=7)
grid_RFC.fit(trainX_res, trainy_res)
best_RFC = grid_RFC.best_estimator_
print('Random_Forest_best', f1_score(trainy, best_RFC.predict(trainX)))
# run_test(RFC_res, 'RFCSMOTE')


xboost_res = xgb.XGBClassifier()
xgb_res_params = dict(gamma=np.arange(0, 10, 1),
                      max_depth=np.arange(2, 10, 1),
                      alpha=np.arange(1, 5, 1))
grid_xgb = GridSearchCV(estimator=xboost_res,
                        param_grid=xgb_res_params,
                        cv=5,
                        scoring='f1',
                        n_jobs=7)
grid_xgb.fit(trainX_res, trainy_res)
best_xgb = grid_xgb.best_estimator_
print('Random Forest_best', f1_score(trainy, best_xgb.predict(trainX)))
# run_test(boost_res, 'boostSMOTE')

params = dict(threshold=np.arange(2.5, 3.5, 0.1)*1e10)
autoencoder = auto_encoder_model(refit=True, Xshape=trainX.shape[1])
grid_autoencoder = GridSearchCV(estimator=autoencoder,
                                param_grid=params,
                                cv=5,
                                scoring='f1')
grid_autoencoder.fit(trainX_res, trainy_res)
best_autoencoder = grid_autoencoder.best_estimator_
print('Autoencoder_best', f1_score(trainy, best_autoencoder.predict(trainX)))

models = [('RFC', best_RFC), ('XGB', best_xgb), ('Autoencoder', best_autoencoder)]
VC = VotingClassifier(estimators=models, voting='hard')