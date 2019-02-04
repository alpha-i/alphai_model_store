from collections import namedtuple
from typing import List

import numpy as np
from alphai_es_datasource.wf.transformers import AbstractTransformer, Splitter
from alphai_es_datasource.utils import PlainChunk
from sklearn import model_selection

RULModelData = namedtuple('RULModelData', 'data labels x_max x_min y_max y_min')


class RULTransformer(AbstractTransformer):
    def __init__(self, X_max=None, X_min=None, y_max=None, y_min=None, reshape_x=True, transpose_x=False,
                 normalise_x=True, normalise_y=True):
        self.x_max = X_max
        self.x_min = X_min
        self.y_max = y_max
        self.y_min = y_min
        self.reshape_x = reshape_x
        self.transpose_x = transpose_x
        self.normalise_x = normalise_x
        self.normalise_y = normalise_y

    def transform(self, data: List[PlainChunk]) -> RULModelData:
        X_ = []
        y_ = []
        drop_cols = ['wind_turbine', 'IsFaulty']

        for i in data:
            X_.append(i.data.drop(columns=drop_cols).values)
            if not i.next_fault_at:
                continue
            tstamp1 = i.starts_at
            tstamp2 = i.next_fault_at
            td = tstamp2 - tstamp1
            td_days = td.total_seconds() / (60 * 60 * 24)
            y_.append(td_days)

        X_ = np.array(X_)
        y_ = np.array(y_)
        print(X_.shape, y_.shape)

        X__ = X_

        if self.reshape_x:
            new_shape = X_.shape[1] * X_.shape[2]
            X__ = X__.reshape((-1, new_shape))

        if self.normalise_x:
            new_shape = X_.shape[1] * X_.shape[2]
            X__ = X__.reshape((-1, new_shape))
            if self.x_max is None:
                self.x_max = np.max(X__, axis=0)
                self.x_min = np.min(X__, axis=0)
            X__ = (X__ - self.x_min) / (self.x_max - self.x_min)
            X__ = X__.reshape((-1, X_.shape[1], X_.shape[2]))

        if self.transpose_x:
            X__ = np.transpose(X__, (0, 2, 1))

        y__ = y_

        print(X__.shape, y__.shape)

        y_max = self.y_max
        y_min = self.y_min
        if self.y_max is None or self.y_min is None:
            if self.normalise_y:
                y_max = np.max(y__, axis=0)
                y_min = np.min(y__, axis=0)

        y__ = (y__ - y_min) / (y_max - y_min)

        return RULModelData(X__, y__, self.x_max, self.x_min, y_max, y_min)


class RULSplitter(Splitter):

    @staticmethod
    def train_test_split(data: RULModelData, test_size=0.2, random_state=1337):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(data.data, data.labels, test_size=test_size,
                                                                            random_state=random_state)
        return (RULModelData(x_train, y_train, data.x_max, data.x_min, data.y_max, data.y_min),
                RULModelData(x_test, y_test, data.x_max, data.x_min, data.y_max, data.y_min))
