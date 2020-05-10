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
        '''
        ['./results/hparams_opt round 13 DTR',
                                               './results/hparams_opt round 13 dtr invariant 10',
                                               './results/hparams_opt round 13 dtr smote 1100',
                                               './results/hparams_opt round 13 ann',
                                               './results/hparams_opt round 13 ann invariant 10',
                                               './results/hparams_opt round 13 ann smote 1100']
        '''

        data_store = prepare_grand_data_store(['./results/hparams_opt round 13 DTR',
                                               './results/hparams_opt round 13 dtr smote 1100',
                                               './results/hparams_opt round 13 dtr invariant 10',
                                               './results/hparams_opt round 13 ann NDA HE',
                                               './results/hparams_opt round 13 ann SMOTE HE',
                                               './results/hparams_opt round 13 ann Invariant HE',])
        hparams = {'init': [0.7, 0.3], 'n_gen':500, 'n_pop':4000, 'eval_func':'eval2'}
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


selector(case=2, write_dir='./results/ga_combination HE NDA+SMOTE+Invariant')
#selector(case=4, write_dir='./results/hparams_opt round 13 ANNs')
