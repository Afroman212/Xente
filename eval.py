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
models = [('RFC', best_RFC), ('XGB', best_xgb), ('Autoencoder', best_autoencoder)]
VC = VotingClassifier(estimators=models, voting='hard')