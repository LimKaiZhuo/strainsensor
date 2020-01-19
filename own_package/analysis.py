import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from own_package.others import print_array_to_excel, print_df_to_excel
from own_package.active_learning.acquisition import load_model_ensemble, model_ensemble_prediction, load_model_chunks
from own_package.features_labels_setup import load_data_to_fl
import os, itertools
from collections import deque

def l2_tracker(write_excel, final_excel_loader, last_idx_store):
    wb = openpyxl.load_workbook(write_excel)
    wb.create_sheet('L2 Results')
    ws = wb[wb.sheetnames[-1]]
    fl = load_data_to_fl(data_loader_excel_file=final_excel_loader, normalise_labels=True,
                         label_type='cutoff',
                         norm_mask=[0, 1, 3, 4, 5])
    final_features = fl.features_c_norm

    suggestions_store = [y2 - y1 for y2, y1 in zip(last_idx_store[1:], last_idx_store[:-1])] + [0]

    batch_l2_store = []
    batch_l2_suggestions_store = []
    for last_idx, suggestions_numel in zip(last_idx_store, suggestions_store):
        features = final_features[:last_idx, :].tolist()

        l2_store = []
        for idx, x in enumerate(features):
            other_features = np.array(features[:idx] + features[idx + 1:])
            l2_distance = np.linalg.norm(x=other_features - np.array(x).reshape((1, -1)), ord=2, axis=1)
            l2_store.append(np.min(l2_distance))
        batch_l2_store.append(np.mean(l2_store))

        if suggestions_numel == 0:
            batch_l2_suggestions_store.append(np.NaN)
        else:
            l2_suggestions_store = []
            suggestions_features = final_features[last_idx:last_idx + suggestions_numel].tolist()
            for sf in suggestions_features:
                l2_distance = np.linalg.norm(x=features - np.array(sf).reshape((1, -1)), ord=2, axis=1)
                l2_suggestions_store.append(np.min(l2_distance))
            batch_l2_suggestions_store.append(np.mean(l2_suggestions_store))

    df = pd.DataFrame(data=np.concatenate((np.array(last_idx_store).reshape(-1, 1),
                                           np.array(batch_l2_store).reshape(-1, 1),
                                           np.array(batch_l2_suggestions_store).reshape(-1, 1)), axis=1),
                      columns=['Expt Batches', 'Mean Min L2', 'Suggestions Mean Min L2'],
                      index=range(1, len(last_idx_store) + 1))
    print_df_to_excel(df=df, ws=ws)
    wb.save(write_excel)


def testset_prediction_results(write_excel, model_dir_store, excel_loader_dir_store, testset_excel_dir, rounds, fn, numel):
    wb = openpyxl.load_workbook(write_excel)
    results_col = fn + 1 + 1 + 2 * numel + 3
    mse_store = []
    mre_store = []
    mare_store = []
    testset_fl = load_data_to_fl(testset_excel_dir, normalise_labels=True, label_type='cutoff', norm_mask=[0, 1, 3, 4, 5])
    column_headers = testset_fl.labels_names
    for idx, (model_dir, loader_excel, round) in enumerate(zip(model_dir_store, excel_loader_dir_store, rounds)):
        wb.create_sheet('Round {}'.format(round))
        ws = wb[wb.sheetnames[-1]]
        fl = load_data_to_fl(data_loader_excel_file=loader_excel, norm_mask=[0, 1, 3, 4, 5],
                             normalise_labels=True, label_type='cutoff')
        model_store = load_model_ensemble(model_dir)
        # Must use the round's fl to scale, not the testset scaler as it might be different
        p_y, _ = model_ensemble_prediction(model_store, fl.apply_scaling(testset_fl.features_c))

        for row, p_label in enumerate(p_y.tolist()):
            if p_label[1] > p_label[2]:
                p_y[row, 1] = p_y[row, 2]
            if p_label[0] > p_y[row, 1]:
                p_y[row, 0] = p_y[row, 1]

        se_store = (testset_fl.labels-p_y) ** 2
        re_store = np.abs(testset_fl.labels-p_y) / testset_fl.labels
        are_store = np.arctan(re_store)

        df = pd.DataFrame(data=np.concatenate((testset_fl.labels, p_y, se_store, re_store, are_store), axis=1),
                          index=list(range(1,1+testset_fl.count)),
                          columns=list(column_headers)
                                  + ['P_{}'.format(col) for col in column_headers]
                                  + ['SE_{}'.format(col) for col in column_headers]
                                  + ['RE_{}'.format(col) for col in column_headers]
                                  + ['ARE_{}'.format(col) for col in column_headers])
        print_df_to_excel(df=df, ws=ws)

        col = fn + 1 + 1 + 2 * numel + 3
        mse_store.append(np.mean(se_store))
        mre_store.append(np.mean(re_store))
        mare_store.append(np.mean(are_store))
        ws.cell(1, col).value = 'MSE'
        ws.cell(1, col + 1).value = mse_store[-1]
        ws.cell(2, col).value = 'MRE'
        ws.cell(2, col + 1).value = mre_store[-1]
        ws.cell(3, col).value = 'ARE'
        ws.cell(3, col + 1).value = mare_store[-1]

    wb.create_sheet('Final_results')
    ws = wb[wb.sheetnames[-1]]
    df = pd.DataFrame(data=np.array([mse_store, mre_store, mare_store]), index=['mse', 're', 'are'], columns=rounds)
    print_df_to_excel(df=df, ws=ws)
    wb.save(write_excel)


def testset_model_results_to_excel(write_excel, model_dir_store, loader_excel, testset_excel_dir, fn,  numel, chunks):
    wb = openpyxl.load_workbook(write_excel)
    mse_store = []
    mre_store = []
    mare_store = []
    testset_fl = load_data_to_fl(testset_excel_dir, normalise_labels=True, label_type='cutoff', norm_mask=[0, 1, 3, 4, 5])
    column_headers = testset_fl.labels_names

    fl = load_data_to_fl(data_loader_excel_file=loader_excel, norm_mask=[0, 1, 3, 4, 5],
                         normalise_labels=True, label_type='cutoff')

    model_name_store = []
    for model_dir in model_dir_store:
        for idx, file in enumerate(os.listdir(model_dir)):
            filename = os.fsdecode(file)
            model_name_store.append(model_dir + '/' + filename)
        print('Loading the following models from {}. Total models = {}'.format(model_dir, len(model_name_store)))
    model_chunks = [model_name_store[x:x+chunks] for x in range(0, len(model_name_store), chunks)]
    testset_features_c_norm = fl.apply_scaling(testset_fl.features_c)
    model_idx = 1
    for single_model_chunk in model_chunks:
        model_store = load_model_chunks(single_model_chunk)
        for idx, model in enumerate(model_store):
            wb.create_sheet('{}'.format(model_idx))
            model_idx += 1
            ws = wb[wb.sheetnames[-1]]
            if model:
                # Must use the round's fl to scale, not the testset scaler as it might be different
                p_y = model.predict(testset_features_c_norm)
                print(np.std(p_y, axis=0))

                for row, p_label in enumerate(p_y.tolist()):
                    if p_label[1] > p_label[2]:
                        p_y[row, 1] = p_y[row, 2]
                    if p_label[0] > p_y[row, 1]:
                        p_y[row, 0] = p_y[row, 1]

                se_store = (testset_fl.labels-p_y) ** 2
                re_store = np.abs(testset_fl.labels-p_y) / testset_fl.labels
                are_store = np.arctan(re_store)

                df = pd.DataFrame(data=np.concatenate((testset_fl.labels, p_y, se_store, re_store, are_store), axis=1),
                                  index=list(range(1,1+testset_fl.count)),
                                  columns=list(column_headers)
                                          + ['P_{}'.format(col) for col in column_headers]
                                          + ['SE_{}'.format(col) for col in column_headers]
                                          + ['RE_{}'.format(col) for col in column_headers]
                                          + ['ARE_{}'.format(col) for col in column_headers])
                print_df_to_excel(df=df, ws=ws)

                col = fn + 1 + 1 + 2 * numel + 3
                mse_store.append(np.mean(se_store))
                mre_store.append(np.mean(re_store))
                mare_store.append(np.mean(are_store))
                ws.cell(1, col).value = 'MSE'
                ws.cell(1, col + 1).value = mse_store[-1]
                ws.cell(2, col).value = 'MRE'
                ws.cell(2, col + 1).value = mre_store[-1]
                ws.cell(3, col).value = 'ARE'
                ws.cell(3, col + 1).value = mare_store[-1]
            else:
                ws.cell(1,1).value = 'EOF error'
                ws.cell(2,1).value = model_name_store[idx-1]
                mse_store.append(np.nan)
                mre_store.append(np.nan)
                mare_store.append(np.nan)

    ws = wb[wb.sheetnames[0]]
    df = pd.DataFrame(data=np.array([mse_store, mre_store, mare_store]).T, columns=['mse', 're', 'are'], index=range(1,1+len(mse_store)))
    df.insert(0, 'Name', model_name_store)
    print_df_to_excel(df=df, ws=ws)
    wb.save(write_excel)


def testset_optimal_combination(write_excel, combination_excel):
    xls = pd.ExcelFile(combination_excel)
    sheetnames = xls.sheet_names[1:]

    df = pd.read_excel(xls, sheetnames[0], index_col=0)

