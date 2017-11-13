""" Contains the predictor class used by the flask app. """
from pandas import read_json
from preprocessing import CompletePreprocessor
from modeling import BaggedAda

df = read_json('data/data.json')
preprocessor = CompletePreprocessor()
X, y = preprocessor.fit_transform(df)

class Predictor(object):
    """ Class that holds the trained model and preprocessor, ready to
    predict the processed data. """
    def __init__(self):
        self.preprocess = preprocessor
        self._model = BaggedAda(n_estimators = 225)
        self._model.fit(X, y)

    def predict(self, data_JSON):
        data_df = read_json(data_JSON)
        X = self.preprocessor.transform(data_df)
        return {'processed data': X,
                'label': self._model.predict(X),
                'probability': self._model.predit_proba(X)}

if __name__ == 'main':
    fitted_model = Predictor()
