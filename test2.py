import numpy as np
import pandas as pd
import openpyxl, pickle, os
from own_package.others import print_df_to_excel, create_excel_file
from own_package.smote.smote_code import produce_smote, create_invariant_testset
from own_package.features_labels_setup import load_data_to_fl
from own_package.data_store_analysis import get_best_trial_from_rounds, get_best_trial_from_rounds_custom_metric
#from own_package.hparam_opt import read_hparam_data


def selector(case, **kwargs):
    if case == 1:
        excel_dir = create_excel_file('./results/smote_data.xlsx')
        fl = load_data_to_fl(
            data_loader_excel_file='./excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13),
            normalise_labels=True,
            label_type='cutoff',
            norm_mask=[0, 1, 3, 4, 5])
        f, l = produce_smote(features=fl.features_c, labels=fl.labels, numel=4000)

        wb = openpyxl.Workbook()
        ws = wb[wb.sheetnames[-1]]
        print_df_to_excel(df=pd.DataFrame(data=np.concatenate((f, l), axis=1),
                                          columns=fl.features_c_names.tolist() + fl.labels_names.tolist()),
                          ws=ws)
        wb.save(excel_dir)
        pass
    elif case == 2:
        testset_excel_dir = './excel/ett_125trainset_points.xlsx'
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=1)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=1)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=1)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=5)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=5)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=5)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=10)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=10)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=10)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=30)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=30)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=30)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=50)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=50)
        create_invariant_testset(testset_excel_dir=testset_excel_dir,
                                 numel=50)
    elif case == 3:
        get_best_trial_from_rounds(dir_store=['./results/hparams_opt round 1 DTR',
                                              './results/hparams_opt round 2 DTR',
                                              './results/hparams_opt round 3 DTR',
                                              './results/hparams_opt round 4 DTR',
                                              './results/hparams_opt round 5 DTR',
                                              './results/hparams_opt round 6 DTR',
                                              './results/hparams_opt round 6e DTR',
                                              './results/hparams_opt round 7 DTR',
                                              './results/hparams_opt round 8 DTR',
                                              './results/hparams_opt round 9 DTR',
                                              './results/hparams_opt round 10 DTR',
                                              './results/hparams_opt round 11 DTR',
                                              './results/hparams_opt round 12 DTR',
                                              './results/hparams_opt round 13 DTR',
                                              './results/hparams_opt round 1 ANN - 2',
                                              './results/hparams_opt round 2 ann - 2',
                                              './results/hparams_opt round 3 ann',
                                              './results/hparams_opt round 4 ann',
                                              './results/hparams_opt round 5 ann',
                                              './results/hparams_opt round 6 ann',
                                              './results/hparams_opt round 6e ann',
                                              './results/hparams_opt round 7 ann',
                                              './results/hparams_opt round 8 ann',
                                              './results/hparams_opt round 9 ann',
                                              './results/hparams_opt round 10 ann',
                                              './results/hparams_opt round 11 ann',
                                              './results/hparams_opt round 12 ann',
                                              './results/hparams_opt round 13 ann'
                                              ],
                                   excel_subname='overall_summary',
                                   sort_col='Val MSE',
                                   results_excel_dir='./results/new_summary.xlsx')
    elif case == 4:
        excel_dir = create_excel_file('./results/new_summary.xlsx')
        get_best_trial_from_rounds_custom_metric(dir_store=[
            ['./results/hparam active learning/hparams_opt round 1 ANN - 2',
             './results/hparam active learning/hparams_opt round 1 DTR_weak_round_1'],
            ['./results/hparam active learning/hparams_opt round 2 ann - 2',
             './results/hparam active learning/hparams_opt round 2 DTR_weak_round_2'],
            ['./results/hparam active learning/hparams_opt round 3 ann',
             './results/hparam active learning/hparams_opt round 3 DTR_weak_round_3'],
            ['./results/hparam active learning/hparams_opt round 4 ann',
             './results/hparam active learning/hparams_opt round 4 DTR_weak_round_4'],
            ['./results/hparam active learning/hparams_opt round 5 ann',
             './results/hparam active learning/hparams_opt round 5 DTR_weak_round_5'],
            ['./results/hparam active learning/hparams_opt round 6 ann',
             './results/hparam active learning/hparams_opt round 6 DTR_weak_round_6'],
            ['./results/hparam active learning/hparams_opt round 6e ann',
             './results/hparam active learning/hparams_opt round 6e DTR_weak_round_6e'],
            ['./results/hparam active learning/hparams_opt round 7 ann',
             './results/hparam active learning/hparams_opt round 7 DTR_weak_round_7'],
            ['./results/hparam active learning/hparams_opt round 8 ann',
             './results/hparam active learning/hparams_opt round 8 DTR_weak_round_8'],
            ['./results/hparam active learning/hparams_opt round 9 ann mse',
             './results/hparam active learning/hparams_opt round 9 DTR_weak_round_9'],
            ['./results/hparam active learning/hparams_opt round 10 ann mse',
             './results/hparam active learning/hparams_opt round 10 DTR_weak_round_10'],
            ['./results/hparam active learning/hparams_opt round 11 ann mse',
             './results/hparam active learning/hparams_opt round 11 DTR_weak_round_11'],
            ['./results/hparam active learning/hparams_opt round 12 ann mse',
             './results/hparam active learning/hparams_opt round 12 DTR_weak_round_12'],
            ['./results/hparam active learning/hparams_opt round 13 ann mse',
             './results/hparam active learning/hparams_opt round 13 DTR_weak_round_13']],
            excel_subname='overall_summary',
            metric_cols=['Train MSE', 'Val MSE'],
            weightage=[0, 1],
            results_excel_dir=excel_dir,
            top_models=3)
    elif case == 4.1:
        excel_dir = create_excel_file('./results/DTR_ANN_NDA_top3_E3.xlsx')
        get_best_trial_from_rounds_custom_metric(dir_store=[
            ['./results/hparams_opt round 13 DTR_weak_NDA_round_13',
             './results/hparams_opt round 13 ann NDA HE',],
        ],
            excel_subname='overall_summary',
            metric_cols=['Train MRE', 'Val MRE'],
            weightage=[0, 1],
            results_excel_dir=excel_dir,
            top_models=3)
    elif case == 5:
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


selector(2, write_dir='./results/hparams_opt round 13 dtr invariant 10')
