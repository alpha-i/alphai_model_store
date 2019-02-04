import datetime

from keras import backend

from alphai_model_store.chunkycnn import ChunkyCNN
from datasource.wf import WFDataSource
from datasource.transformers import ModelInfo, SimpleTransformer, Splitter

# Data specifications
component = 'global'
start_date = datetime.datetime(2015, 8, 8)
end_date = datetime.datetime(2016, 2, 28)
turbine_list = range(1, 11)


model_info = ModelInfo(yaml_file='./tests/model_info.yaml')

wf_datasource = WFDataSource(
        host='51.144.39.71:9200',
        index_name='wf_scada_hist',
        start_date=datetime.datetime(2015, 8, 8),
        end_date=datetime.datetime(2016, 2, 28),
        chunk_size=72,
        stride=1,
        abnormal_window_duration=datetime.timedelta(hours=96),
        turbines=(1,),
        fields=model_info.all_variables()
    )

all_data = wf_datasource.get_all_data()
transformed_model_data = SimpleTransformer(reshape_x=False, transpose_x=False, normalise_x=True).transform(all_data)
train_data, test_data = Splitter.train_test_split(transformed_model_data, test_size=0.2, random_state=1337)

# # Get data from turbines specified in turbine list
# all_data = get_model_data(turbine_list, component, start_date, end_date,
#                           chunk_size=72, stride=1, abnormal_window=96, incl_ambient=False)
#
# # Data pre-processing as per model requirements
# X, y = data_transform(all_data, reshape_x=False, transpose_x=False, normalise_x=True)
#
# X_tra, X_tst, y_tra, y_tst = train_test_split(X, y, test_size=0.2, random_state=1337)
#
# y_tra = np_utils.to_categorical(y_tra)
# y_tst = np_utils.to_categorical(y_tst)

print(train_data.data[train_data.data == 0].shape,
      train_data.labels[train_data.labels == 1].shape,
      test_data.data[test_data.data == 0].shape,
      test_data.labels[test_data.labels == 1].shape)

# model input shape must be specified before initialisation of the class
input_dim = train_data.data[0].shape
backend.clear_session()
convnet = ChunkyCNN(data_shape=input_dim, model_config_dir='model_dir', component_name='global')
convnet.train(train_data, n_epochs=50)

convnet.load_model_weights('global_modelparams.h5')
prediction = convnet.predict(test_data)
convnet.evaluate(test_data, prediction)
