import logging
import numpy as np
from sklearn.ensemble import IsolationForest

class iForest():
    MAX_N_SAMPLES = 32000

    def __init__(self, max_number_of_samples=None, outliers_fraction=0.1, n_estimators=100):
        self.max_number_of_samples = max_number_of_samples if max_number_of_samples else self.MAX_N_SAMPLES
        self.outliers_fraction = outliers_fraction
        self.n_estimators = n_estimators
        self.classifier = IsolationForest(n_estimators=self.n_estimators,max_samples=self.max_number_of_samples,
                                          contamination=self.outliers_fraction, random_state=None)

    def train(self, train_data):

        n_train_samples = train_data.shape[0]
        train_data = train_data.reshape(n_train_samples, -1)

        if n_train_samples > self.max_number_of_samples:
            logging.warning(
                'Discarding training data: using {} of {} chunks.'.format(self.max_number_of_samples, n_train_samples))
            train_data = self._subsample_data(train_data)

        self.classifier.fit(train_data)

    def predict(self, test_sample):
        data = test_sample.reshape(test_sample.data.shape[0], -1)
        prediction = self.classifier.predict(data)
        return prediction

    def decision_function(self, test_sample):
        data = test_sample.data.reshape(test_sample.data.shape[0], -1)
        anomaly_score = self.classifier.decision_function(data)
        return np.squeeze(anomaly_score)

    def _subsample_data(self, data):
        return data[np.random.choice(data.shape[0], self.max_number_of_samples, replace=False)]

    @property
    def configuration(self):
        return {
            'max_number_of_samples': self.max_number_of_samples,
            'outliers_fraction': self.outliers_fraction,
            'n_estimators': self.n_estimators
        }
