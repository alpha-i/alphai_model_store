import logging
from collections import namedtuple
from collections.__init__ import namedtuple
from typing import List

import numpy as np
from alphai_es_datasource.utils import Chunk

from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from alphai_model_store.chunkycnn import ChunkyCNN

from keras import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, BatchNormalization, MaxPooling1D, LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *

from alphai_model_store.transformers.rul import RULModelData

MODEL_FILENAME_TEMPLATE = '{}/{}_modelparams.h5'

RULPrediction = namedtuple("Prediction", "results")
EvaluationStats = namedtuple("EvaluationStats", "mse")

class ChunkyCNNRUL(ChunkyCNN):
    def _build_model(self, input_dimension):
        """
        Builds the network graph for the ChunkyCNN model.
        :param input_dimension: tuple, e.g. (72, 97) i.e. (time, sensors).
        """
        model = Sequential()
        model.add(Conv1D(input_shape=input_dimension, filters=128, kernel_size=10))
        model.add(LeakyReLU(alpha=.001))
        model.add(MaxPooling1D(pool_size=(1), strides=(1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=128, kernel_size=10))
        model.add(LeakyReLU(alpha=.001))
        model.add(MaxPooling1D(pool_size=(2), strides=(1)))
        # model.add(Conv1D(filters=128, kernel_size=3))
        # model.add(LeakyReLU(alpha=.001))
        # model.add(MaxPooling1D(pool_size=(3), strides=(1)))
        # model.add(BatchNormalization())
        # model.add(Conv1D(filters=128, kernel_size=3))
        # model.add(LeakyReLU(alpha=.001))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=.001))
        model.add(BatchNormalization())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=.001))
        model.add(BatchNormalization())
        model.add(Dense(1))
        model.add(LeakyReLU(alpha=.001))

        return model

    def train(self, data: RULModelData, n_epochs=25, batch_size=100):
        """
        Public method to either train a new model or retrain a saved model.
        :param X_train, Array of shape (-1,sensors,time) or (-1,time,sensors).
        :param y_train, Array of categorical labels, of shape (-1, num_classes). In our case, num_classes=2.
        :param n_epochs: integer, number of complete passes through the training data.
        :param batch_size: integer, number of training points in each iteration within an epoch.
        """

        # Will be better to enforce a date directory
        model_filename = MODEL_FILENAME_TEMPLATE.format(self.model_config_dir, self.component_name)

        rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        self.model.compile(loss='mse', optimizer=rmsprop, metrics=['mse'])

        cb = [ModelCheckpoint(filepath=model_filename, monitor='val_loss', save_best_only=True)]
        self.model.fit(data.data, data.labels, validation_split=0.125, callbacks=cb, epochs=n_epochs, batch_size=batch_size,
                       verbose=True)

    def predict(self, data: RULModelData) -> RULPrediction:
        """
        Public method that returns the probability that a chunk is anomalous, and the predicted class for that chunk.
        Must be preceded by load_model().
        :param X_test: Array of shape (-1,sensors,time) or (-1,time,sensors). Unseen data.
        :return: pred_prob[:,1] : An array of anomaly probabilities; pred_class : array of predicted classes.
        """
        X_test = data.data
        rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        self.model.compile(loss='mse', optimizer=rmsprop, metrics=['mse'])
        rul = self.model.predict(X_test)

        return RULPrediction(results=rul)

    def evaluate(self, test_data: RULModelData, prediction: RULPrediction):
        """
        Private method for evaluating the performance of the model on unseen data. Used for experimentation.
        Assumes self.predict() has been run.
        :param X_test: Array of shape (-1,sensors,time) or (-1,time,sensors). Unseen data.
        """
        X_test = test_data.data
        y_test = test_data.labels
        test_loss, test_mse = self.model.evaluate(X_test, y_test)
        logging.info('Test loss : ', test_loss)
        logging.info('Test mse : ', test_mse)
        logging.info('\n')
        return EvaluationStats(mse=test_mse)


RULModelPrediction = namedtuple('RULModelPrediction', 'component timestamp rul')


class RULPredictionList(list):
    def __init__(self, component: str, data: List, prediction: RULPrediction):
        self._data = data
        self._component = component
        self._prediction = prediction
        self._timestamps = self.calculate_prediction_points(self._data)
        self.prediction_with_timestamps = zip(self._timestamps, self._prediction.results.flatten())
        self._inner_list = []
        for timestamp, rul in self.prediction_with_timestamps:
            self._inner_list.append(
                RULModelPrediction(self._component, timestamp, rul),
            )

        super().__init__(self._inner_list)

    def calculate_prediction_points(self, chunks: List[Chunk]):
        timestamps = []
        for chunk in chunks:
            start = chunk.data.iloc[0].name
            end = chunk.data.iloc[-1].name
            duration = end - start
            half_duration = duration / 2
            half_point = start + half_duration
            timestamps.append(half_point)

        return timestamps
