from typing import List

import numpy as np
from alphai_es_datasource.utils import Chunk, Metadata, ModelData

from alphai_model_store.transformers.base import AbstractTransformer


class SimpleTransformer(AbstractTransformer):
    DROP_COLS = ['wind_turbine', 'label', 'IsFaulty']

    def __init__(self, reshape_x=True, transpose_x=False, normalise_x=True):
        self.reshape_x = reshape_x
        self.transpose_x = transpose_x
        self.normalise_x = normalise_x

    def transform(self, data: List[Chunk], metadata: Metadata = None) -> ModelData:
        X_ = []
        y_ = []

        # TODO
        mean = 0
        std = 0

        for i in data:
            X_.append(i.data.drop(columns=self.DROP_COLS).values)
            y_.append(i.label.value)

        X_ = np.array(X_)
        y_ = np.array(y_)

        logging.debug(f"Shape at at 110: {X_.shape}, {y_.shape}")
        X__ = X_

        if self.reshape_x:
            new_shape = X_.shape[1] * X_.shape[2]
            X__ = X__.reshape((-1, new_shape))

        if self.normalise_x:
            new_shape = X_.shape[1] * X_.shape[2]
            X__ = X__.reshape((-1, new_shape))
            if not metadata:
                mean = np.mean(X__, axis=0)
                std = np.std(X__, axis=0)
            else:
                mean = metadata.mean
                std = metadata.std
            X__ = (X__ - mean) / std
            X__ = X__.reshape((-1, X_.shape[1], X_.shape[2]))

        if self.transpose_x:
            X__ = np.transpose(X__, (0, 2, 1))

        y__ = y_

        return ModelData(data=X__, labels=y__, metadata=Metadata(mean=mean, std=std))
