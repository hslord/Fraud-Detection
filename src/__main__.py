<<<<<<< HEAD
import pandas as pd
from numpy import nan_to_num
import matplotlib.pyplot as plt


from modeling import *
from preprocessor import DFTransformer
from preprocessing import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_json('data/data.json')

df_transformer = DFTransformer()
df_transformer.fit(df)

#df = df_transformer.transform(df)
df = preprocessing(df)

"""
y_label = 'acct_type'
X_labels = df_transformer.features_
X_labels.remove(y_label)

## Hardcoded errors
X_labels.remove('payee_name')
X_labels.remove('venue_name')
X_labels.remove('listed')
"""

all_labels = df.columns.tolist()
y_label = 'acct_type'
X_labels = all_labels
X_labels.remove(y_label)

y = df[y_label].astype(float).values     
X = df[X_labels].astype(float).values  
X = nan_to_num(X)

def try_bmodels():
    string = "BAG: {}, ADA: {}"
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
        clh = BaggedAda(n_estimators = 150)
        clh.fit(X_train, y_train)
        clm = AdaCustom(n_estimators = 50)
        clm.fit(X_train, y_train)
        print string.format(clh.score(X_test, y_test),
                            clm.score(X_test, y_test))

def metrics_bag():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
    clm = BaggedAda(n_estimators = 250)
    clf = AdaCustom(n_estimators = 200)
    clm.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    print calc_scores(clm, X_test, y_test)
    bag_roc_a, bag_roc_b, _ = clm.plot_roc(X_test, y_test, False)
    ada_roc_a, ada_roc_b, _ = clf.plot_roc(X_test, y_test, False)
    plt.plot(bag_roc_a, bag_roc_b)
    plt.plot(ada_roc_a, ada_roc_b)
    plt.show()

def plot_adarounds():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
    rounds = range(0, 5) + range(5, 75, 5) + range(75, 525, 25)
    train_loss = []
    test_loss = []
    models = []
    for n_rounds in rounds:
        clf = AdaCustom(n_estimators = n_rounds + 1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        train_loss.append(metrics.log_loss(y_train, y_pred_train))
        test_loss.append(metrics.log_loss(y_test, y_pred))
    plt.plot(rounds, train_loss, label = 'Training Loss')
    plt.plot(rounds, test_loss, label = 'Testing Loss')
    plt.xlabel('boosting_rounds')
    plt.ylabel('loss')
    plt.title('Log loss over more rounds of boosting')
    plt.legend(loc = 'upper right')
    plt.show()

def gridsearch_bag():
    params = {'n_estimators': [215, 220, 225, 230, 235, 240]}
    clf = GridSearchCV(BaggedAda(), params)
    clf.fit(X, y)
    return clf

def gridsearch_ada():
    params = {'n_estimators': [50, 100,150, 200, 250]}
    clf= GridSearchCV(AdaCustom(), params)
    clf.fit(X, y)
    return clf

n_models = plot_adarounds()
=======
""" Area for testing models. """
from modeling import estimator
import pandas as pd

df = pd.read_json('data/data.json')
>>>>>>> 3874f957a7bb697f8857a5e6621d32e0e9b98c1f
