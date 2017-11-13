""" Two Boosted Models. """
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator as SklearnBE
from sklearn import metrics

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

def calc_score(y_pred, y_true):
    return metrics.roc_auc_score(y_pred, y_true)

def calc_scores(model, X, y):
    y_true = y
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return [metrics.roc_auc_score(y_true, y_prob),
            metrics.f1_score(y_pred, y_true),
            metrics.accuracy_score(y_pred, y_true),
            metrics.precision_score(y_pred, y_true),
            metrics.recall_score(y_pred, y_true)]

def check_classes(y):
    """ Never use. Unless the model is broken. """
    if y.sum() == 0:
        y[0] = True
    return y

class AdaCustom(SklearnBE):
    def __init__(self, **params):
        self._classifier = GradientBoostingClassifier(loss = 'deviance',
                                                      **params)
    def fit(self, X, y):
        return self._classifier.fit(X, y)

    def predict(self, X):
        return self._classifier.predict(X)

    def predict_proba(self, X):
        return self._classifier.predict_proba(X)[:,1]

    def set_params(self, **params):
        self.__init__(**params)
        return self

    def score(self, X, y):
        y_pred = self.predict_proba(X)
        return calc_score(y, y_pred)

    def plot_roc(self, X, y, plot = False):
        y_prob = self.predict_proba(X)
        plot_data = metrics.roc_curve(y, y_prob)
        if plot:
            A, B, thresholds = plot_data
            plt.plot(A, B)
            plt.show()
        return metrics.roc_curve(y, y_prob)

class BaggedAda(AdaCustom):
    def __init__(self, **params):
        self._classifier = BaggingClassifier(
            GradientBoostingClassifier(loss = 'exponential',
                                       **params),
            n_estimators = 5)


    
