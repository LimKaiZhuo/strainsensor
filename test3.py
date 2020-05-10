import numpy as np
from tensorflow.python.keras import backend as K

import pickle, os

from own_package.hparam_opt import read_hparam_data


def selector(case, **kwargs):
    if case == 1:
        def np_haitao_error(y_true, y_pred):
            diff = np.abs((y_true - y_pred) / np.reshape(np.abs(y_true[:,-1]), (-1,1)))
            return 100. * np.mean(diff, axis=-1)

        def haitao_error(y_true, y_pred):
            diff = K.abs((y_true - y_pred) / K.reshape(K.clip(K.abs(y_true[:,-1]),
                                                    K.epsilon(),
                                                    None), (-1,1)))
            return 100. * K.mean(diff, axis=-1)

        y_true = np.array([[4.5,5,7.5],
                           [3,9,29]])
        y_pred = np.array([[2.757726, 6.397033, 9.976769],
                           [4.774668, 10.33038, 36.04665]])

        print(np_haitao_error(y_true, y_pred))
        print(K.eval(haitao_error(K.variable(y_true), K.variable(y_pred))))
    elif case == 2:
        ett_names = ['I01-1', 'I01-2', 'I01-3',
                     'I05-1', 'I05-2', 'I05-3',
                     'I10-1', 'I10-2', 'I10-3',
                     'I30-1', 'I30-2', 'I30-3',
                     'I50-1', 'I50-2', 'I50-3',
                     '125Test', '125Test I01', '125Test I05','125Test I10']
        write_dir = kwargs['write_dir']
        data_store = []
        for filename in os.listdir(write_dir):
            if filename.endswith(".pkl"):
                with open('{}/{}'.format(write_dir, filename), 'rb') as handle:
                    data_store.extend(pickle.load(handle))
        read_hparam_data(data_store=data_store, write_dir=write_dir, ett_names=ett_names, print_s_df=False,
                         trainset_ett_idx=-4)
        pass

selector(2, write_dir='./results/hparams_opt round 13 ann mse')
