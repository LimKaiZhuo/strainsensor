import numpy as np
import pandas as pd
import openpyxl
from tensorflow.python.keras import backend as K

import pickle, os

from own_package.hparam_opt import read_hparam_data
from own_package.others import print_df_to_excel, create_excel_file


def selector(case, **kwargs):
    if case == 1:
        def np_haitao_error(y_true, y_pred):
            diff = np.abs((y_true - y_pred) / np.reshape(np.abs(y_true[:, -1]), (-1, 1)))
            return 100. * np.mean(diff, axis=-1)

        def haitao_error(y_true, y_pred):
            diff = K.abs((y_true - y_pred) / K.reshape(K.clip(K.abs(y_true[:, -1]),
                                                              K.epsilon(),
                                                              None), (-1, 1)))
            return 100. * K.mean(diff, axis=-1)

        y_true = np.array([[4.5, 5, 7.5],
                           [3, 9, 29]])
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
                     '125Test', '125Test I01', '125Test I05', '125Test I10']
        write_dir = kwargs['write_dir']
        data_store = []
        for filename in os.listdir(write_dir):
            if filename.endswith(".pkl"):
                with open('{}/{}'.format(write_dir, filename), 'rb') as handle:
                    data_store.extend(pickle.load(handle))
        read_hparam_data(data_store=data_store, write_dir=write_dir, ett_names=ett_names, print_s_df=False,
                         trainset_ett_idx=-4)
        pass
    elif case == 3:
        # Name checking for data_store files in various folders
        dir_store = ['./results/hparams_opt round 13 ann NDA HE',
                     './results/hparams_opt round 13 ann Invariant HE',

                     './results/hparams_opt round 13 dtr invariant 10',
                     './results/hparams_opt round 13 DTR',
                     ]

        data_store = []
        for dir in dir_store:
            for filename in os.listdir(dir):
                if filename.endswith(".pkl"):
                    with open('{}/{}'.format(dir, filename), 'rb') as handle:
                        data = pickle.load(handle)
                        data_store.append([dir, data[0][0][0][0]])
                    break
        excel_dir = create_excel_file('./results/read_data_store_names.xlsx')
        wb = openpyxl.load_workbook(excel_dir)
        ws = wb[wb.sheetnames[-1]]
        df = pd.DataFrame(data_store)
        print_df_to_excel(df=df, ws=ws)
        wb.save(excel_dir)

#for i in [13,]:
#    selector(case=2, write_dir='./results/hparams_opt round {} DTR_weak_I50b_round_{}'.format(i, i))
#selector(case=3, write_dir='./results/test')
selector(case=2, write_dir='./results/hparams_opt round 1 conv1_round_1')