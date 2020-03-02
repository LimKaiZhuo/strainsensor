import numpy as np
import pandas as pd
import openpyxl
from own_package.others import print_df_to_excel, create_excel_file
from own_package.smote.smote_code import produce_smote, create_invariant_testset
from own_package.features_labels_setup import load_data_to_fl
from own_package.data_store_analysis import get_best_trial_from_rounds, get_best_trial_from_rounds_custom_metric


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
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=1)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=1)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=1)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=5)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=5)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=5)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=10)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=10)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=10)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=30)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=30)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=30)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=50)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
                                 numel=50)
        create_invariant_testset(testset_excel_dir='./excel/ett_30testset_cut.xlsx',
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
        get_best_trial_from_rounds_custom_metric(dir_store=['./results/hparams_opt round 1 DTR',
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
                                   metric_cols=['Train MSE','Val MSE'],
                                                 weightage=[0.3, 0.7],
                                   results_excel_dir='./results/new_summary.xlsx')

selector(4)
