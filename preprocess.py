import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


class preprocess_data:
    def __init__(self,
                 training=None,
                 test=None):

        self.sm = SMOTE(sampling_strategy=0.50, random_state=42, k_neighbors=3)
        self.training = training
        self.testing = test
        self.train = None
        self.Tid_test = None
        self.X_test = None

    def preprocess_train(self):
        df = pd.read_csv(self.training)
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

        Tid = df['TransactionId'].to_frame()
        X = proc_df.drop(['FraudResult'], axis=1)
        y = proc_df['FraudResult']
        X_res, y_res = self.sm.fit_resample(X, y)
        X_res = pd.DataFrame(data=X_res, columns=X.columns)
        self.train = (Tid, X, y, X_res, y_res)
        return Tid, X, y, X_res, y_res

    def preprocess_test(self):
        df = pd.read_csv(self.training)
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

        Tid = df['TransactionId'].to_frame()
        Tid_test = df['TransactionId'].to_frame()
        X_test = proc_df
        return Tid_test, X_test

    def run_test(self, model, name):
        Tid_test, X_test = self.preprocess_test()
        fraud = model.predict(X_test)
        test = Tid_test
        test['FraudResult'] = fraud
        test.to_csv('test_submission_%s.csv' % name, index=False)
