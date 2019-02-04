import logging
import numpy as np
from sklearn import svm

class SVC():
    MAX_N_SAMPLES = 32000

    def __init__(self, max_number_of_samples=None, C=1.0, epsilon=0.1):
        self.max_number_of_samples = max_number_of_samples if max_number_of_samples else self.MAX_N_SAMPLES
        self.epsilon = epsilon
        self.C = C
        self.classifier = svm.SVC(C = self.C, kernel="rbf", gamma='auto')

    def train(self, train_data, train_labels):

        n_train_samples = train_data.shape[0]
        train_data = train_data.reshape(n_train_samples, -1)

        if n_train_samples > self.max_number_of_samples:
            logging.warning(
                'Discarding training data: using {} of {} chunks.'.format(self.max_number_of_samples, n_train_samples))
            train_data = self._subsample_data(train_data)

        self.classifier.fit(train_data, train_labels)

    def predict(self, test_sample):
        data = test_sample.reshape(test_sample.shape[0], -1)
        prediction = self.classifier.predict(data)
        return prediction

    def decision_function(self, test_sample):
        data = test_sample.reshape(test_sample.shape[0], -1)
        anomaly_score = self.classifier._decision_function(data)
        return np.squeeze(anomaly_score)

    def _subsample_data(self, data):
        return data[np.random.choice(data.shape[0], self.max_number_of_samples, replace=False)]

    @property
    def configuration(self):
        return {
            'max_number_of_samples': self.max_number_of_samples,
            'epsilon': self.epsilon
        }

class OneClassSVM():
    MAX_N_SAMPLES = 32000

    def __init__(self, max_number_of_samples=None, nu=0.1):
        self.max_number_of_samples = max_number_of_samples if max_number_of_samples else self.MAX_N_SAMPLES
        self.nu = nu
        self.classifier = svm.OneClassSVM(nu=self.nu, kernel="rbf", gamma=0.1)

    def train(self, train_sample):
        train_data = train_sample.data

        n_train_samples = train_data.shape[0]
        train_data = train_data.reshape(n_train_samples, -1)

        if n_train_samples > self.max_number_of_samples:
            logging.warning(
                'Discarding training data: using {} of {} chunks.'.format(self.max_number_of_samples, n_train_samples))
            train_data = self._subsample_data(train_data)

        self.classifier.fit(train_data)

    def predict(self, test_sample):
        data = test_sample.data.reshape(test_sample.data.shape[0], -1)
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
            'nu': self.nu
        }
