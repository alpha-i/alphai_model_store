import logging
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, roc_auc_score
from sklearn import decomposition
from .__init__ import AbstractNeuralNetwork

from keras import models, regularizers, Sequential, callbacks
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, BatchNormalization


class FFNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, train_data, validation_data, test_data):
        self.train_data = train_data
        self.val_data = validation_data
        self.test_data = test_data

    def train(self, n_epochs, reweight=False):
        X_tra, y_tra = data_transform(self.train_data)
        X_val, y_val = data_transform(self.val_data)
        X_test, y_test = data_transform(self.test_data)

        # Perform PCA on data
        pca_comp_data = 10
        X_tra_data, \
        X_val_data, \
        X_tst_data = get_model_data(X_tra, X_val, X_test,
                                    data_version='pca', pca_components=pca_comp_data)

        # Normalise data using mean and std of training data before input to neural network
        X_tra_data, \
        X_val_data, \
        X_tst_data = get_model_data(X_tra_data, X_val_data, X_tst_data,
                                    data_version='normalised')

        input_dim = X_tra_data.shape[1]
        self.norm_test_data = X_tst_data
        self.test_labels = y_test
        self.model = get_ffnn(input_dim)

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        cb = callbacks.EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=200,
                                       verbose=0, mode='auto')
        if reweight is False:
            self.model.fit(X_tra_data, y_tra, validation_data=(X_val_data,y_val),
                           callbacks=[cb], epochs=n_epochs)
        elif reweight is True:
            cw = {0: 1., 1: 8}
            self.model.fit(self.train_data, y_tra, validation_data=(self.val_data,y_val),
                           callbacks=[cb], epochs=n_epochs, class_weight=cw)

    def retrain(self):
        pass

    def predict(self):
        pred_prob = self.model.predict(self.norm_test_data)
        pred_class = np.argmax(pred_prob, 1)
        self.pred_prob = pred_prob
        self.pred_class = pred_class

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.norm_test_data, self.test_labels)
        print('Test loss : ', test_loss)
        print('Test accuracy : ', test_acc)
        print('ROC AUC score : ', roc_auc_score(self.test_labels, self.pred_prob[:, 1]))
        print('\n')
        print(classification_report(self.test_labels, self.pred_class))

    @property
    def configuration(self):
        return {
            'model':  self.model,
            'pred_prob':  self.pred_prob,
            'pred_class':  self.pred_class
        }


def get_ffnn(input_dimension):
    model = Sequential()
    model.add(Dense(256, activation=tf.nn.relu, input_dim=input_dimension))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation=tf.nn.softmax))

    return model


def data_transform(data):
    X_ = []
    y_ = []
    drop_cols = ['wind_turbine', 'label', 'IsFaulty']

    for indx, i in enumerate(data):
        X_.append(i.data.drop(columns=drop_cols).values)
        y_.append(i.label.value)

    X_ = np.array(X_)
    y_ = np.array(y_)

    new_shape = X_.shape[1] * X_.shape[2]
    X__ = X_.reshape((-1, new_shape))
    y__ = y_.reshape((-1, 1))

    return X__, y__


def get_model_data(X_tra, X_val, X_test, data_version='original', pca_components=10):
    if data_version == 'original':
        X_tra_data = X_tra
        X_val_data = X_val
        X_tst_data = X_test
    elif data_version == 'normalised':
        mean = np.mean(X_tra, axis=0)
        std = np.std(X_tra, axis=0)
        X_tra_data = (X_tra - mean) / std
        X_val_data = (X_val - mean) / std
        X_tst_data = (X_test - mean) / std
    elif data_version == 'pca':
        pca = decomposition.KernelPCA(n_components=pca_components, kernel='rbf')
        pca.fit(X_tra)
        X_tra_data = pca.transform(X_tra)
        X_val_data = pca.transform(X_val)
        X_tst_data = pca.transform(X_test)

    return X_tra_data, X_val_data, X_tst_data