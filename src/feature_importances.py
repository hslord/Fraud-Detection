from modeling import *
from preprocessing import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Get data into X/y train_test_split format
df = pd.read_json('../data/data.json')

df = preprocessing(df)
y = df.pop('acct_type')
X = df
features = X.columns
X = np.nan_to_num(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Fit GBT Model
GBT = AdaCustom().fit(X_train, y_train)

# Get most important features from GBT Model
importances = GBT.feature_importances_
GBT_top_features = features[importances > np.mean(importances)]
GBT_top_feature = features[importances == np.max(importances)]
