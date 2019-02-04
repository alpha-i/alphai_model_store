import abc
from typing import List

from alphai_es_datasource.utils import Chunk, ModelData


class AbstractTransformer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def transform(self, data: List[Chunk]) -> ModelData:
        """
        Transforms a list of chunks into a piece of data that the model can use.
        """
        raise NotImplementedError
