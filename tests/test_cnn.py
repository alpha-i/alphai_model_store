import datetime
import logging

import numpy as np
import pytest
from alphai_es_datasource.wf import WFDataSource
from alphai_es_datasource.wf.transformers import SimpleTransformer, ModelInfo, Splitter
from alphai_es_datasource.utils import ModelPredictionList, Labels
from keras import backend
from keras.utils.layer_utils import count_params

from alphai_model_store.chunkycnn import ChunkyCNN

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)
logging.getLogger('urllib3.connectionpool').setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@pytest.fixture('session')
def model_info():
    return ModelInfo(yaml_file='./tests/model_info.yaml')


@pytest.fixture('session')
def datasource(model_info):
    wf_datasource = WFDataSource(
        host='51.144.39.71:9200',
        index_name='wf_scada_hist',
        start_date=datetime.datetime(2015, 8, 8),
        end_date=datetime.datetime(2016, 2, 28),
        chunk_size=72,
        stride=1,
        abnormal_window_duration=96,
        turbines=(1,),
        fields=model_info.all_variables()
    )

    return wf_datasource


@pytest.fixture('session')
def raw_data(datasource):
    return datasource.get_all_data()


@pytest.fixture('session')
def test_data(datasource):
    return datasource.get_test_data()


@pytest.fixture('session')
def simple_transformer():
    return SimpleTransformer(reshape_x=False, transpose_x=False, normalise_x=True)


def test_model_implementation_should_look_like(raw_data, simple_transformer):
    transformed_model_data = simple_transformer.transform(raw_data)
    train_data, test_data = Splitter.train_test_split(transformed_model_data)

    backend.clear_session()
    convnet = ChunkyCNN(
        data_shape=(72, 63),
        model_config_dir='./tests/model_dir',
        component_name='global'
    )
    assert count_params(convnet.model.trainable_weights) == 32414

    convnet.train(train_data, n_epochs=1)
    layer_0_weights = convnet.model.layers[0].get_weights()
    convnet.load_model_weights('global_modelparams.h5')
    convnet.train(train_data, n_epochs=1)
    assert not np.array_equal(layer_0_weights, convnet.model.layers[0].get_weights())

    convnet.load_model_weights('global_modelparams.h5')
    prediction = convnet.predict(test_data)
    assert all(np.round(prediction.probabilities) == prediction.classes)

    evaluation_stats = convnet.evaluate(test_data, prediction)
    assert np.sum(evaluation_stats.support) == len(test_data.labels)


def test_model_prediction_with_timestamps(raw_data, test_data, simple_transformer):
    # training should contain a mix of normal/abnormal bits,
    # otherwise the splitter won't be able to balance them
    trainable_data = raw_data[:500] + raw_data[-500:]
    transformed_model_data = simple_transformer.transform(trainable_data)
    train_data, _ = Splitter.train_test_split(transformed_model_data)

    testable_data = test_data[5:10]  # it should contain 5 chunks

    # we need to transform the test data using the same metadata
    # that we got from the original training data
    # (this needs to be saved somehow in the platform)
    # (because it doesn't really belong to the model)
    test_data = simple_transformer.transform(testable_data, metadata=transformed_model_data.metadata)

    backend.clear_session()
    convnet = ChunkyCNN(
        data_shape=(72, 63),
        model_config_dir='./tests/model_dir',
        component_name='global'
    )
    convnet.train(train_data, n_epochs=1)

    prediction = convnet.predict(test_data)

    full_prediction = ModelPredictionList(
        component='global',
        data=testable_data,
        prediction=prediction
    )

    # we must get as many predictions as data chunks
    assert len(full_prediction) == 5

    # test that the prediction point is in the middle
    assert testable_data[0].data.index[0] < full_prediction[0].timestamp < testable_data[0].data.index[-1]

    # test that we get an abnormality probability between 0 and 1
    assert 0 <= full_prediction[0].abnormality <= 1.0

    # we must get a label of normal or abnormal
    assert full_prediction[0].prediction in (Labels.NORMAL, Labels.ABNORMAL)
