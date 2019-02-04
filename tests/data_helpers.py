import os

import numpy as np
import tensorflow as tf
import yaml
from alphai_es_datasource.wf.datasource import WFDataSource
from alphai_es_datasource.wf.definitions import WF_COLUMNS, WF_EXTRA_COLUMNS, WF_STRING_COLUMNS

from tests.definitions import WF_COLUMNS, WF_EXTRA_COLUMNS, WF_STRING_COLUMNS

dirname = os.path.dirname(__file__)
MODEL_YAML = os.path.join(dirname, 'model_info.yaml')

tf.logging.set_verbosity(tf.logging.INFO)


def get_model_data_with_rul(turbine_list, component, start_date, end_date,
                            chunk_size=144, stride=144, abnormal_window=240, incl_ambient=False):
    model_vars = get_model_vars(model=component, incl_ambient=incl_ambient)
    all_data = []
    for turbine in turbine_list:
        print(turbine, ': ', end='')

        chunks = get_turbine_data_with_rul(model_vars,
                                           start_date=start_date,
                                           end_date=end_date,
                                           turbine=turbine,
                                           chunk_size=chunk_size,
                                           stride=stride,
                                           abnormal_window=240)

        all_data += chunks[turbine]

    np.random.shuffle(all_data)
    return all_data


def get_model_data(turbine_list, component, start_date, end_date,
                   chunk_size=72, stride=1, abnormal_window=96, incl_ambient=False):
    model_vars = get_model_vars(model=component, incl_ambient=incl_ambient)
    all_data = []
    for turbine in turbine_list:
        print(turbine, ': ', end='')

        train_data, \
        val_data, \
        test_data = get_turbine_data_with_rul(model_vars,
                                              start_date=start_date,
                                              end_date=end_date,
                                              turbine=turbine,
                                              chunk_size=72,
                                              stride=1,
                                              abnormal_window=96)

        all_data += train_data + val_data + test_data

    np.random.shuffle(all_data)
    return all_data


def get_turbine_data_with_rul(model_vars,
                              start_date,
                              end_date,
                              turbine=1,
                              chunk_size=72,
                              stride=1,
                              abnormal_window=96):
    wf_data = WFDataSource(
        host='51.144.39.71:9200',
        index_name='wf_scada_hist',
        start_date=start_date,
        end_date=end_date,
        chunk_size=chunk_size,
        stride=stride,
        abnormal_window_duration=abnormal_window,
        turbines=(turbine,),
        fields=tuple(model_vars),
    )

    return wf_data.get_list_of_chunks()


def get_model_vars(model='global',
                   incl_ambient=False,
                   all_vars=WF_COLUMNS,
                   extra_vars=WF_EXTRA_COLUMNS,
                   string_vars=WF_STRING_COLUMNS,
                   model_yaml=MODEL_YAML):
    '''
    Load model variables for different component models
    - ambient (PCA -> arches)
    - drive_train (PCA -> arches)
    - nacelle_yaw
    - power_features
    - rotor_hub
    - tower
    - turbine_performance
    - other (all other variables)
    '''

    model_info = yaml.load(open(model_yaml))

    if model == 'global':
        # get all the variables?
        model_vars = set(WF_COLUMNS) - set(WF_EXTRA_COLUMNS) - set(WF_STRING_COLUMNS)
        model_vars = sorted(list(model_vars))
    else:
        # only get the relevant
        # TODO: incorporate this into the YAML file somehow
        if incl_ambient:
            ambient_vars = model_info['ambient']['variables']
            model_vars = model_info[model]['variables']
            model_vars += ambient_vars
        else:
            model_vars = model_info[model]['variables']

    return model_vars


def data_transform(data, reshape_x=True, transpose_x=False, normalise_x=True):
    X_ = []
    y_ = []
    drop_cols = ['wind_turbine', 'label', 'IsFaulty']

    for i in data:
        X_.append(i.data.drop(columns=drop_cols).values)
        y_.append(i.label.value)

    X_ = np.array(X_)
    y_ = np.array(y_)
    print(X_.shape, y_.shape)

    X__ = X_

    if reshape_x:
        new_shape = X_.shape[1] * X_.shape[2]
        X__ = X__.reshape((-1, new_shape))

    if normalise_x:
        new_shape = X_.shape[1] * X_.shape[2]
        X__ = X__.reshape((-1, new_shape))
        mean = np.mean(X__, axis=0)
        std = np.std(X__, axis=0)
        X__ = (X__ - mean) / std
        X__ = X__.reshape((-1, X_.shape[1], X_.shape[2]))

    if transpose_x:
        X__ = np.transpose(X__, (0, 2, 1))

    y__ = y_

    print(X__.shape, y__.shape)

    return X__, y__
