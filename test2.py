import numpy as np
import pandas as pd
import openpyxl
from own_package.others import print_df_to_excel, create_excel_file
from own_package.smote.smote_code import produce_smote
from own_package.features_labels_setup import load_data_to_fl


def selector(case, **kwargs):
    if case == 1:
        excel_dir = create_excel_file('./results/smote_data.xlsx')
        fl = load_data_to_fl(data_loader_excel_file='./excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13),
                             normalise_labels=True,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        f, l = produce_smote(features=fl.features_c, labels=fl.labels, numel=4000)

        wb = openpyxl.Workbook()
        ws = wb[wb.sheetnames[-1]]
        print_df_to_excel(df=pd.DataFrame(data=np.concatenate((f,l), axis=1),
                                          columns=fl.features_c_names.tolist()+fl.labels_names.tolist()),
                          ws=ws)
        wb.save(excel_dir)
        pass


selector(1)
