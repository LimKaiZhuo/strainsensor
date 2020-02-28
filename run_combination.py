import numpy as np
import pickle
from own_package.combination_2 import ga_train_val_eval_on_test, prepare_grand_data_store
from own_package.others import create_results_directory
from own_package.hparam_opt import read_hparam_data


def selector(case, **kwargs):
    if case == 1:
        write_dir = kwargs['write_dir']
        file_name = '{}/data_store.pkl'.format(write_dir)
        # Load data (deserialize)
        with open(file_name, 'rb') as handle:
            data_store = pickle.load(handle)
        pass
    elif case == 2:
        write_dir = kwargs['write_dir']
        data_store = prepare_grand_data_store(['./results/hparams_opt round 13 DTR Original ett - 2',
                                               ])
        hparams = {'init': [0.7, 0.3], 'n_gen':1200, 'n_pop':2000, 'eval_func':'eval1'}
        results_dir = create_results_directory(write_dir)
        ga_train_val_eval_on_test(results_dir=results_dir, data_store=data_store, hparams=hparams)
    elif case==4:
        write_dir = kwargs['write_dir']
        file_name= '{}/data_store.pkl'.format(write_dir)
        # Load data (deserialize)
        with open(file_name, 'rb') as handle:
            data_store = pickle.load(handle)
        read_hparam_data(data_store=data_store, write_dir=write_dir)
        pass


selector(case=2, write_dir='./results/ga_combination DTR ns+s+i + ANN ns+s')
#selector(case=4, write_dir='./results/hparams_opt round 13 ANNs')
