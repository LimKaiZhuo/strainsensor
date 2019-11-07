import keras.backend as K
import tensorflow as tf
import gc
import numpy as np
import pandas as pd
from pandas import Series
import openpyxl
from openpyxl import load_workbook
from sklearn.metrics import mean_squared_error
import time, os
# Own Scripts
from own_package.models.models import MTmodel, Kmodel, Pmodel
from own_package.models.hul_model import HULMTmodel
from .others import print_array_to_excel
from .features_labels_setup import load_data_to_fl

def run_skf(model_mode, loss_mode, fl, fl_store, hparams, norm_mask, normalise_labels,labels_norm,
            skf_file,
            skf_sheet=None,
            k_folds=10, k_shuffle=True, save_model=False, save_model_name=None, save_model_dir=None,
            plot_name=None):
    '''
    Stratified k fold cross validation for training and evaluating model 2 only. Model 1 data is trained before hand.
    :param model_mode: Choose between using SNN or cDNN (non_smiles) and SNN_smiles or cDNN_smiles
    :param cv_mode: Cross validation mode. Either 'skf' or 'loocv'.
    :param hparams: hparams dict containing hyperparameters information
    :param loader_file: data_loader excel file location
    :param skf_file: skf_file name to save excel file as
    :param skf_sheet: name of sheet to save inside the skf_file excel. If None, will default to SNN or cDNN as name
    :param k_folds: Number of k folds. Used only for skf cv_mode
    :param k_shuffle: Whether to shuffle the given examples to split into k folds if using skf
    :return:
    '''

    # Run k model instance to perform skf
    predicted_labels_store = []
    mse_store = []
    mse_norm_store = []
    folds = []
    val_idx = []
    val_features_c = []
    val_labels = []
    for fold, fl_tuple in enumerate(fl_store):
        sess = tf.Session()
        K.set_session(sess)
        instance_start = time.time()
        (ss_fl, i_ss_fl) = fl_tuple  # ss_fl is training fl, i_ss_fl is validation fl

        # Set up model
        if loss_mode == 'normal':
            model = MTmodel(fl=ss_fl, mode=model_mode, hparams=hparams, labels_norm=labels_norm)
        elif loss_mode == 'hul':
            model = HULMTmodel(fl=ss_fl, mode=model_mode, hparams=hparams, labels_norm=labels_norm)
            print('HUL Standard Deviation Values:')
            print([np.exp(K.get_value(log_var[0])) ** 0.5 for log_var in model.model.layers[-1].log_vars])
        elif loss_mode == 'ann':
            model = Kmodel(fl=ss_fl, mode=model_mode, hparams=hparams, labels_norm=labels_norm)
        elif loss_mode == 'p_model':
            model = Pmodel(fl=ss_fl, mode=model_mode, hparams=hparams, labels_norm=labels_norm)
        else:
            raise KeyError('loss_mode ' + loss_mode + 'is not a valid selection for loss mode.')

        # Train model and save model training loss vs epoch plot if plot_name is given, else no plot will be saved
        if plot_name:
            model.train_model(ss_fl, i_ss_fl, save_mode=False,
                              plot_name='{}_fold_{}.png'.format(plot_name, fold))
        else:
            model.train_model(ss_fl, i_ss_fl)

        # Evaluation
        predicted_labels, mse, mse_norm = model.eval(i_ss_fl)
        if fl.normalise_labels:
            predicted_labels_store.extend(fl.labels_scaler.inverse_transform(predicted_labels))
        else:
            predicted_labels_store.extend(predicted_labels)
        mse_store.append(mse)
        mse_norm_store.append(mse_norm)
        '''
        if fold == k_folds-1:
            stringlist = []
            model.model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = "\n".join(stringlist)
            print(short_model_summary)
        '''
        # Saving model
        if save_model:
            # Set save_model_name
            if isinstance(save_model_name, str):
                save_model_name1 = save_model_name + '_' + model_mode + '_' + str(fold + 1)
            else:
                save_model_name1 = model_mode + '_' + str(fold + 1)

            # Save model
            print('Saving instance {} model in {}'.format(fold + 1, save_model_dir + save_model_name1 + '.h5'))
            if loss_mode == 'normal' or loss_mode == 'ann':
                model.model.save(save_model_dir + save_model_name1 + '.h5')
            elif loss_mode == 'hul':
                model.prediction_model.save(save_model_dir + save_model_name1 + '.h5')

        # Need to put the next 3 lines if not memory will run out
        del model
        K.clear_session()
        sess.close()
        gc.collect()

        # Preparing data to put into new_df that consists of all the validation dataset and its predicted labels
        folds.extend([fold] * i_ss_fl.count)  # Make a col that contains the fold number for each example
        if len(val_features_c):
            val_features_c = np.concatenate((val_features_c, i_ss_fl.features_c),
                                            axis=0)
        else:
            val_features_c = i_ss_fl.features_c

        val_labels.extend(i_ss_fl.labels)
        val_idx.extend(i_ss_fl.idx)

        # Printing one instance summary.
        instance_end = time.time()
        print(
            '\nFor k-fold run {} out of {}. Each fold has {} examples. Model is {} with {} loss. Time taken for '
            'instance = {}\n'
            'Post-training results: \nmse = {}, mse_norm = {}\n'
            '####################################################################################################'
                .format(fold + 1, k_folds, i_ss_fl.count, model_mode, loss_mode, instance_end - instance_start, mse,
                        mse_norm))

    mse_avg = np.average(mse_store)
    mse_norm_avg = np.average(mse_norm_store)

    # Calculating metrics based on complete validation prediction
    mse_full = mean_squared_error(val_labels, predicted_labels_store)
    try:
        mse_norm_full = mean_squared_error(fl.labels_scaler.transform(val_labels),
                                           fl.labels_scaler.transform(predicted_labels_store))
    except AttributeError:
        mse_norm_full=mse_full

    # Creating dataframe to print into excel later.
    new_df = np.concatenate((np.array(folds)[:, None],  # Convert 1d list to col. vector
                             val_features_c,
                             np.array(val_labels),
                             np.array(predicted_labels_store))
                            , axis=1)
    predicted_labels_name = list(map(str, np.arange(2,21)))
    predicted_labels_name = ['P_' + x for x in predicted_labels_name]
    headers = ['folds'] + \
              list(map(str, fl.features_c_names)) + \
              list(map(str, np.arange(2,21))) + \
              predicted_labels_name

    # val_idx is the original position of the example in the data_loader
    new_df = pd.DataFrame(data=new_df, columns=headers, index=val_idx)

    print('Writing into' + skf_file)
    wb = load_workbook(skf_file)

    # Creating new worksheet. Even if SNN worksheet already exists, a new SNN1 ws will be created and so on
    if skf_sheet is None:
        wb.create_sheet(model_mode)
    else:
        wb.create_sheet(model_mode + skf_sheet)
    sheet_name = wb.sheetnames[-1]  # Taking the ws name from the back ensures that if SNN1 is the new ws, it works

    # Writing hparam dataframe first
    pd_writer = pd.ExcelWriter(skf_file, engine='openpyxl')
    pd_writer.book = wb
    pd_writer.sheets = dict((ws.title, ws) for ws in wb.worksheets)
    new_df.to_excel(pd_writer, sheet_name)
    start_col = len(new_df.columns) + 4
    hparams = pd.DataFrame(dict([(k, Series(v)) for k, v in hparams.items()]))
    hparams.to_excel(pd_writer, sheet_name, startrow=0, startcol=start_col - 1)
    start_row = 5

    # Writing other subset split, instance per run, and bounds
    ws = wb[sheet_name]
    headers = ['mse', 'mse_norm']
    values = [mse_avg, mse_norm_avg]
    values_full = [mse_full, mse_norm_full]
    print_array_to_excel(np.array(headers), (1 + start_row, start_col + 1), ws, axis=1)
    print_array_to_excel(np.array(values), (2 + start_row, start_col + 1), ws, axis=1)
    print_array_to_excel(np.array(values_full), (3 + start_row, start_col + 1), ws, axis=1)
    ws.cell(2 + start_row, start_col).value = 'Folds avg'
    ws.cell(3 + start_row, start_col).value = 'Overall'
    pd_writer.save()
    pd_writer.close()
    wb.close()

    return mse_norm_full
