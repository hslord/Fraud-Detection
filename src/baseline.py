import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class Baseline(object):
    """
    Blank Model for baseline models. Functions left intentionally blank.
    Initializes with train and test data.
    """
    def __init__(self):
        self.fitted = False
        self.score_fn = {'accuracy': accuracy_score,
                       'recall': recall_score,
                       'precision': precision_score,
                       'f1_score': f1_score}

    def fit(self, X, y):
        self.fitted=True

    def predict(self, X):
        return None

    def score(self, y_true, y_pred, scoring=['accuracy', 'recall', 'precision', 'f1_score']):
        """
        Calculates the accuracy of the model (#correctly classified/
        #incorrectly classified)
        INPUTS:
        y_pred (numpy array) - array of predicted labels of shape [n_samples, 1]
        y (numpy array) - array of actual labels of shape [n_samples, 1]
        scoring (string) - currently nonfunctional. Future implentation planned
                        for different accuracy metrics.
        OUTPUTS:
        acc (float) - accuracy of the model, as defined by the scoring parameter
        """
        if not self.fitted:
            print("Model has not been fitted.")
            return None

        acc = {}
        for score_type in scoring:
            acc[score_type] = self.score_fn[score_type](y_true, y_pred)
        return acc


class WeightedGuess(Baseline):
    """
    Looks at the distribution of labels in the train data, and makes a random
    guess in proportion with the label distribution (e.g. if 40% of the labels
    in the train data are 'A', the model will guess A 40% of the time on average)
    """
    def fit(self, X, y):
        """
        Stores the unique label values as a list. Stores the relative proportion
        of each label as corresponding list.
        INPUTS:
        X - feature matrix. Traditional input to fit function, but unused here.
        y (1-dimensional pandas dataframe) - series containing labels
        """
        proportions = 1.0*y.value_counts()/y.value_counts().sum()
        self.labels = proportions.index.values.astype(bool)
        self.thresholds = proportions.values
        self.fitted = True

    def predict(self,X):
        """
        Makes a random guess in proportion to the label distribution
        INPUTS:
        X (pandas dataframe or numpy array) - feature matrix of shape
                                            [n_samples, n_features]
        OUTPUT:
        y_pred (numpy array) - predicted label matrix of shape [n_features, 1]
        """
        y_pred = np.random.choice(self.labels, size=(X.shape[0],), p=self.thresholds)
        return y_pred


class MajorityGuess(WeightedGuess):
    """
    Looks at the distribution of labels in the train data, and guess the most
    frequently occurring label (e.g. if 60% of the labels in the train data are
    'A', the model will guess A every time).
    """
    def fit(self, X, y):
        """
        Stores the unique label values as a list. Finds the label with the
        highest relative frequency
        INPUTS:
        X - feature matrix. Traditional input to fit function, but unused here.
        y (1-dimensional pandas dataframe) - series containing labels
        """
        proportions = y.value_counts()/y.value_counts().sum()
        self.labels = proportions.index.values.astype(bool)
        self.guess = np.argmax(proportions)
        self.fitted = True

    def predict(self, X):
        """
        Makes a random guess in proportion to the label distribution
        INPUTS:
        X (pandas dataframe or numpy array) - feature matrix of shape
                                            [n_samples, n_features]
        OUTPUT:
        y_pred (numpy array) - predicted label matrix of shape [n_features, 1]
        """
        y_pred = np.full(shape=(X.shape[0],), fill_value=self.guess)
        return y_pred


def create_baselines(X_train, y_train, X_test, y_test, score_types=['accuracy', 'recall', 'precision', 'f1_score']):
    """Function to instantiate, fit, and score the baseline models for the given
    data. Used to benchmark other models.
    INPUTS:
    X_train (pandas dataframe) - train features matrix of shape
                                [n_samples, n_features]
    y_train (pandas dataframe) - train labels matrix of shape [n_samples, 1]
    X_test (pandas dataframe) - test features matrix of shape
                                [x_samples, n_features]
    y_test (pandas dataframe) - test labels matrix of shape [x_samples, 1]
    scorers (list) - list of strings, keywords to the accuracy metrics desired
    OUTPUTS:
    baselines (list of floats) - list containing accuracy scores for each
                                baseline model in order of: WeightedGuess,
                                MajorityGuess
    """
    # establish baseline models
    print("Running baseline models...")
    baselines = {'Weighted Random Guess': WeightedGuess(), 'Guess Most Frequent': MajorityGuess()}
    baseline_scores = {}
    for name in baselines:
        model = baselines[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = model.score(y_test, y_pred, scoring=score_types)
        baseline_scores[name] = scores
        print("{} Scores: ".format(name))
        for metric in scores:
            print("{} score: {}".format(metric.capitalize(), round(scores[metric], 5)))
    return baselines, baseline_scores


def create_basic_classifiers(X_train, y_train, X_test, y_test, score_types=['accuracy', 'recall', 'precision', 'f1_score']):
    # establish basic classifiers
    print("\nRunning basic classifiers...")
    score_fn = {'accuracy': accuracy_score,
               'recall': recall_score,
               'precision': precision_score,
               'f1_score': f1_score}
    basics = {'GaussianNB': GaussianNB(), 'Logistic Regression': LogisticRegression(solver='newton-cg', n_jobs=-1)}
    basic_scores = {}
    for name in basics:
        model = basics[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("{} Scores: ".format(name))
        scores = {}
        for metric in score_types:
            score = score_fn[metric](y_test, y_pred)
            scores[metric] = score
            print("{} score : {}".format(metric.capitalize(), round(score, 5)))
        basic_scores[name] = scores
    return basics, basic_scores


def run_basic_models(X_train, y_train, X_test, y_test, score_types=['accuracy', 'recall', 'precision', 'f1_score']):
    """
    Helper function to create, fit and score baseline models and basic classifiers on the data.
    INPUTS:
    X_train (pandas dataframe) - train features matrix of shape
                                [n_samples, n_features]
    y_train (pandas dataframe) - train labels matrix of shape [n_samples, 1]
    X_test (pandas dataframe) - test features matrix of shape
                                [x_samples, n_features]
    y_test (pandas dataframe) - test labels matrix of shape [x_samples, 1]
    OUTPUTS:
    models (list of dictionaries) - list of dict, each containing keys of the model name, values of the fitted models
    scores (list of dictionaries) - list of dict, each containing keys of the model name, values of dicts of scoring metric:value
    """
    baselines, baseline_scores = create_baselines(X_train, y_train, X_test, y_test, score_types=score_types)
    basics, basic_scores = create_basic_classifiers(X_train, y_train, X_test, y_test, score_types=score_types)
    models = {'baselines': baselines, 'basic classifiers': basics}
    scores = {'baselines': baseline_scores, 'basic classifiers': basic_scores}
    return models, scores

if __name__ == '__main__':

    # a test run with a 1 column X (which gives GNB fits)
    df = pd.read_json('../data/data.json')
    subset = df[:100]
    X,y = subset.gts, subset.acct_type == 'fraudster'
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    X_train, X_test  = X_train.reshape((len(X_train),1)), X_test.reshape((len(X_test),1))

    m, s = run_basic_models(X_train,y_train,X_test,y_test)
