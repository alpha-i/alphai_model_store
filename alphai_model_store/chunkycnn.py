import logging
import os

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

from alphai_model_store.basemodel import AbstractModel
from alphai_es_datasource.utils.dataclasses import ModelData, Prediction, EvaluationStats

MODEL_FILENAME_TEMPLATE = '{}/{}_modelparams.h5'


class ChunkyCNN(AbstractModel):
    def __init__(self, data_shape, model_config_dir, component_name):
        """
        Instantiates a ChunkyCNN and transforms the data as per the input specifications of the model.

        :param data_shape: tuple, e.g. (72, 97) i.e. (time, sensors), needed to determine input shape of model.
        :param model_config_dir: Directory to save model parameters/neural network weights.
        :param component_name: string, the component model name.
                - ambient
                - drive_train
                - nacelle_yaw
                - power_features
                - rotor_hub
                - tower
                - turbine_performance
                - other
        """
        self.load_model = False
        self.component_name = component_name
        self.model_config_dir = os.path.abspath(model_config_dir)

        # create the model directory if it doesn't already exist...
        logging.debug(f"Creating model configuration folder in {self.model_config_dir}")
        os.makedirs(self.model_config_dir, exist_ok=True)

        self.model = self._build_model(data_shape)

    def load_model_weights(self, load_filename):
        """
        # Allows the user to load a pre-trained model.

        :param load_filename: File with stored model parameters/neural network weights.
        """
        self.load_model = True
        self.load_filename = load_filename
        assert self.load_filename is not None, 'No load_filename specified.'

        model_filename = '{}/{}'.format(self.model_config_dir, self.load_filename)
        self.model.load_weights(model_filename)

    def _build_model(self, input_dimension):
        """
        Builds the network graph for the ChunkyCNN model.
        :param input_dimension: tuple, e.g. (72, 97) i.e. (time, sensors).
        """
        model = Sequential()
        model.add(Conv1D(input_shape=input_dimension, activation='relu', filters=32, kernel_size=10))
        model.add(BatchNormalization())
        model.add(Conv1D(activation='relu', filters=16, kernel_size=10))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(2, activation=tf.nn.softmax))
        model.summary()

        return model

    def train(self, data: ModelData, n_epochs=25, batch_size=100):
        """
        Public method to either train a new model or retrain a saved model.
        :param X_train, Array of shape (-1,sensors,time) or (-1,time,sensors).
        :param y_train, Array of categorical labels, of shape (-1, num_classes). In our case, num_classes=2.
        :param n_epochs: integer, number of complete passes through the training data.
        :param batch_size: integer, number of training points in each iteration within an epoch.
        """
        X_train = data.data
        y_train = data.labels

        # Will be better to enforce a date directory
        model_filename = MODEL_FILENAME_TEMPLATE.format(self.model_config_dir, self.component_name)

        rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

        cb = [ModelCheckpoint(filepath=model_filename, monitor='val_acc', save_best_only=True)]
        self.model.fit(X_train, y_train, validation_split=0.125, callbacks=cb, epochs=n_epochs, batch_size=batch_size,
                       verbose=True)

    def predict(self, data: ModelData) -> Prediction:
        """
        Public method that returns the probability that a chunk is anomalous, and the predicted class for that chunk.
        Must be preceded by load_model().

        :param data: Contains a .data array of shape (-1,sensors,time) or (-1,time,sensors). Unseen data.
        :return: (An array of anomaly probabilities, array of predicted classes)
        """

        X_test = data.data
        rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
        pred_prob = self.model.predict(X_test)
        pred_class = np.argmax(pred_prob, 1)

        return Prediction(probabilities=pred_prob[:, 1], classes=pred_class)

    def evaluate(self, test_data: ModelData, prediction: Prediction) -> EvaluationStats:
        """
        Private method for evaluating the performance of the model on unseen data. Used for experimentation.
        Assumes self.predict() has been run.

        :param test_data: Unseen data.
        """

        X_test = test_data.data
        y_test = test_data.labels
        test_loss, test_acc = self.model.evaluate(X_test, y_test)

        logging.info('Test loss: {}'.format(test_loss))
        logging.info('Test accuracy: {}'.format(test_acc))
        logging.info('ROC AUC score: {}'.format(roc_auc_score(y_test[:, 1], prediction.probabilities)))

        report = classification_report(y_test[:, 1], prediction.classes, target_names=['NORMAL', 'ABNORMAL'])
        logging.info(report)
        p, r, f1, s = precision_recall_fscore_support(y_test[:, 1], prediction.classes, labels=[0, 1])
        return EvaluationStats(precision=p, recall=r, f1_score=f1, support=s)
