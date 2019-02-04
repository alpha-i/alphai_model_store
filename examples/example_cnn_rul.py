import datetime

from keras import backend

from alphai_model_store.chunkycnn_rul import ChunkyCNNRUL
from alphai_model_store.transformers.rul import RULTransformer, RULSplitter
from tests.data_helpers import get_model_data_with_rul

# Data specifications
component = 'global'
start_date = datetime.datetime(2015, 8, 8)
end_date = datetime.datetime(2016, 2, 28)

# Get data from turbines specified in turbine list
turbine_list = tuple(range(1, 3))

chunks = get_model_data_with_rul(turbine_list, component, start_date, end_date,
                                 chunk_size=144, stride=144, abnormal_window=240, incl_ambient=False)

# Data pre-processing as per model requirements
rul_transformer = RULTransformer(reshape_x=False, transpose_x=False, normalise_x=True)
transformed_data = rul_transformer.transform(chunks)
########## normalise y
train_data, test_data = RULSplitter.train_test_split(transformed_data, test_size=0.2, random_state=1337)

# print(y_tra[y_tra==0].shape, y_tra[y_tra==1].shape, y_tst[y_tst==0].shape, y_tst[y_tst==1].shape)

# model input shape must be specified before initialisation of the class
input_dim = train_data.data[0].shape
backend.clear_session()
convnet = ChunkyCNNRUL(data_shape=input_dim, model_config_dir='model_dir', component_name='global')
convnet.train(train_data, n_epochs=1)

convnet.load_model_weights('global_modelparams.h5')
prediction = convnet.predict(test_data)
convnet.evaluate(test_data, prediction)

# Denormalised results
pred_denorm = prediction.results * (transformed_data.y_max - transformed_data.y_min) + transformed_data.y_min
import ipdb; ipdb.set_trace()
