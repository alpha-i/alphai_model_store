import datetime

import pytest
from alphai_es_datasource.wf import WFDataSource
from alphai_es_datasource.wf.transformers import ModelInfo

from alphai_model_store.chunkycnn_rul import ChunkyCNNRUL, RULPredictionList, RULPrediction
from alphai_model_store.transformers.rul import RULTransformer, RULSplitter


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
        chunk_size=144,
        stride=144,
        abnormal_window_duration=240,
        turbines=(1,),
        fields=model_info.all_variables()
    )

    return wf_datasource


@pytest.fixture('session')
def raw_data(datasource):
    train_data, validation_data, test_data = datasource.get_list_of_chunks()
    return train_data, validation_data, test_data


@pytest.fixture('session')
def test_data(datasource):
    return datasource.get_test_data()


@pytest.fixture('session')
def rul_transformer():
    return RULTransformer(reshape_x=False, transpose_x=False, normalise_x=True, normalise_y=True)


def test_model_implementation_should_look_like(datasource, rul_transformer):
    all_data = datasource.get_list_of_chunks()[1]
    train_data = all_data[:500] + all_data[-500:]
    testable_data = all_data[5:500]
    test_data = rul_transformer.transform(testable_data)

    transformed_data = rul_transformer.transform(train_data)  # because we only have 1 turbine?
    train_data, _ = RULSplitter.train_test_split(transformed_data, test_size=0.2, random_state=1337)

    input_dim = train_data.data[0].shape
    convnet = ChunkyCNNRUL(data_shape=input_dim, model_config_dir='./model_dir', component_name='global')
    convnet.train(train_data, n_epochs=1)
    convnet.load_model_weights('global_modelparams.h5')
    prediction = convnet.predict(test_data)
    convnet.evaluate(test_data, prediction)
    prediction_denormalised = prediction.results * (test_data.y_max - test_data.y_min) + test_data.y_min
    prediction_denormalised = RULPrediction(results=prediction_denormalised)

    predictions = RULPredictionList('global', data=testable_data, prediction=prediction_denormalised)
    assert predictions
