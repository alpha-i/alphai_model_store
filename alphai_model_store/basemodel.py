from abc import abstractmethod, ABCMeta


class AbstractModel(metaclass=ABCMeta):

    @abstractmethod
    def load_model_weights(self, load_filename):
        """
        Loads model weights from a specified filename.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_model(self, input_dimension):
        """
        Builds the model.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, data, n_epochs, batch_size):
        """
        Performs the training of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        """
        Performs the prediction of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_data, prediction):
        """
        Evaluates the performance of the model. Used for experimentation.
        """
        raise NotImplementedError
