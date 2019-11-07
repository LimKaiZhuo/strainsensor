
import pandas as pd
import numpy as np
import gc, pickle, time
from openpyxl import load_workbook
from skopt import gp_minimize, dummy_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from sklearn.metrics import matthews_corrcoef, mean_squared_error
# Own Scripts
from own_package.models.models import create_hparams
from own_package.svm_classifier import run_classification, SVMmodel
from own_package.svr import SVRmodel, run_svr
from .cross_validation import run_skf
from own_package.others import print_array_to_excel

def hparam_opt(model_mode, loss_mode, norm_mask, normalise_labels,labels_norm, fl_in, fl_store_in, write_dir, save_model_dir,
               total_run, instance_per_run=3,
               plot_dir=None):
    """
     names = ['shared_1_l', 'shared_1_h',
              'shared_2_l', 'shared_2_h',
              'ts_1_l', 'ts_1_h',
              'ts_2_l', 'ts_2_h',
              'epochs_l', 'epochs_h',
              'l1_l', 'l1_h']
    :param model_mode:
    :param loader_file:
    :param total_run:
    :param instance_per_run:
    :param hparam_file:
    :return:
    """

    hparam_file = write_dir + '/skf_results.xlsx'

    global run_count, data_store, fl, fl_store
    run_count = 0
    fl = fl_in
    fl_store = fl_store_in


    if model_mode == 'hps':
        bounds = [[5, 200,],
                  [0, 200,],
                  [5, 200,],
                  [0, 200,],
                  [10, 2000],
                  [0, 0.3]]
        shared_1 = Integer(low=bounds[0][0], high=bounds[0][1], name='shared_1')
        shared_2 = Integer(low=bounds[1][0], high=bounds[1][1], name='shared_2')
        ts_1 = Integer(low=bounds[2][0], high=bounds[2][1], name='ts_1')
        ts_2 = Integer(low=bounds[3][0], high=bounds[3][1], name='ts_2')
        epochs = Integer(low=bounds[4][0], high=bounds[4][1], name='epochs')
        reg_l1 = Real(low=bounds[5][0], high=bounds[5][1], name='reg_l1')
        dimensions = [shared_1, shared_2, ts_1, ts_2, epochs, reg_l1]
        default_parameters = [30, 30, 30, 30, 300, 0.01]


        @use_named_args(dimensions=dimensions)
        def fitness(shared_1, shared_2, ts_1, ts_2, epochs, reg_l1):
            global run_count, best_loss, data_store, fl, best_hparams
            run_count += 1

            hparams = create_hparams(shared_layers=[shared_1, shared_2], ts_layers=[ts_1, ts_2],epochs=epochs,reg_l1=reg_l1,
                                     verbose=0)

            mse_avg = 0


            for cnt in range(instance_per_run):
                if plot_dir:
                    plot_name = '{}/{}_{}_{}'.format(plot_dir, model_mode, loss_mode, cnt)
                else:
                    plot_name = None
                mse = run_skf(model_mode=model_mode, loss_mode=loss_mode, cv_mode='skf', hparams=hparams,
                              norm_mask=norm_mask, labels_norm=labels_norm,
                              loader_file=loader_file,
                              skf_file=hparam_file, skf_sheet='_' + str(run_count) + '_' + str(cnt),
                              k_folds=10, k_shuffle=True,
                              save_model_name='_' + str(run_count) + '_' + str(cnt), save_model=True,
                              save_model_dir='./save/models',
                              plot_name=plot_name)
                mse_avg += mse

            mse_avg = mse_avg / instance_per_run
            if mse_avg < best_loss:
                best_hparams = hparams
                best_loss = mse_avg
            loss = mse_avg
            print('**************************************************************************************************\n'
                  'Run Number {} \n'
                  'Instance per run {} \n'
                  'Current run MSE {} \n'
                  '*********************************************************************************************'.format(
                run_count, instance_per_run, mse_avg))
            print(pd.DataFrame(hparams))
            print()
            return loss
    elif model_mode == 'cs':
        bounds = [[5, 200, ],
                  [5, 200, ],
                  [0, 200, ],
                  [10, 2000],
                  [0, 0.3]]
        shared_1 = Integer(low=bounds[0][0], high=bounds[0][1], name='shared_1')
        shared_2 = Integer(low=bounds[1][0], high=bounds[1][1], name='shared_2')
        shared_3 = Integer(low=bounds[2][0], high=bounds[2][1], name='shared_3')
        epochs = Integer(low=bounds[3][0], high=bounds[3][1], name='epochs')
        reg_l1 = Real(low=bounds[4][0], high=bounds[4][1], name='reg_l1')
        dimensions = [shared_1, shared_2, shared_3, epochs, reg_l1]
        default_parameters = [30, 30, 30, 300, 0.01]



        @use_named_args(dimensions=dimensions)
        def fitness(shared_1, shared_2, shared_3, epochs, reg_l1):
            global run_count, best_loss, data_store, fl, best_hparams
            run_count += 1

            hparams = create_hparams(cs_layers=[shared_1, shared_2, shared_3], epochs=epochs,
                                     reg_l1=reg_l1,
                                     verbose=0)

            mse_avg = 0

            for cnt in range(instance_per_run):
                if plot_dir:
                    plot_name = '{}/{}_{}_run_{}_count_{}'.format(plot_dir, model_mode, loss_mode,run_count, cnt)
                else:
                    plot_name = None
                mse = run_skf(model_mode=model_mode, loss_mode=loss_mode, cv_mode='skf', hparams=hparams,
                              norm_mask=norm_mask, labels_norm=labels_norm,
                              loader_file=loader_file,
                              skf_file=hparam_file, skf_sheet='_' + str(run_count) + '_' + str(cnt),
                              k_folds=10, k_shuffle=True,
                              save_model_name='_' + str(run_count) + '_' + str(cnt), save_model=True,
                              save_model_dir='./save/models',
                              plot_name=plot_name)
                mse_avg += mse

            mse_avg = mse_avg / instance_per_run
            if mse_avg < best_loss:
                best_hparams = hparams
                best_loss = mse_avg
            loss = mse_avg
            print('**************************************************************************************************\n'
                  'Run Number {} \n'
                  'Instance per run {} \n'
                  'Current run MSE {} \n'
                  '*********************************************************************************************'.format(
                run_count, instance_per_run, mse_avg))
            print(pd.DataFrame(hparams))
            print()
            return loss

    elif model_mode == 'conv1':
        start_time = time.time()
        bounds = [[50, 400, ],
                  [1, 50, ],
                  [100, 600]]

        pre = Integer(low=bounds[0][0], high=bounds[0][1], name='pre')
        filters = Integer(low=bounds[1][0], high=bounds[1][1], name='filters')
        epochs = Integer(low=bounds[2][0], high=bounds[2][1], name='epochs')
        dimensions = [pre, filters, epochs]
        default_parameters = [70, 1, 500]

        @use_named_args(dimensions=dimensions)
        def fitness(pre, filters, epochs):
            global run_count, data_store, fl, fl_store
            run_count += 1
            hparams = create_hparams(pre=pre, filters=filters, epochs=epochs,
                                     reg_l1=0.05, reg_l2=0.05,
                                     verbose=0)

            mse_avg = 0

            for cnt in range(instance_per_run):
                if plot_dir:
                    plot_name = '{}/{}_{}_run_{}_count_{}'.format(plot_dir, model_mode, loss_mode,run_count, cnt)
                else:
                    plot_name = None
                mse = run_skf(model_mode=model_mode, loss_mode=loss_mode, fl=fl, fl_store=fl_store, hparams=hparams,
                              norm_mask=norm_mask, normalise_labels=normalise_labels, labels_norm=labels_norm,
                              skf_file=hparam_file, skf_sheet='_' + str(run_count) + '_' + str(cnt),
                              k_folds=10, k_shuffle=True,
                              save_model_name='hparam_' + str(run_count) + '_' + str(cnt+1), save_model=True,
                              save_model_dir=save_model_dir,
                              plot_name=plot_name)
                mse_avg += mse

            mse_avg = mse_avg / instance_per_run
            loss = mse_avg
            end_time = time.time()
            print('**************************************************************************************************\n'
                  'Run Number {} \n'
                  'Instance per run {} \n'
                  'Current run MSE {} \n'
                  'Time Taken: {}\n'
                  '*********************************************************************************************'.format(
                run_count, instance_per_run, mse_avg, end_time-start_time))
            return loss
    elif model_mode == 'ann3':
        start_time = time.time()
        bounds = [[50, 300, ],
                  [100, 900]]

        pre = Integer(low=bounds[0][0], high=bounds[0][1], name='pre')
        epochs = Integer(low=bounds[1][0], high=bounds[1][1], name='epochs')
        dimensions = [pre, epochs]
        default_parameters = [70,  500]

        @use_named_args(dimensions=dimensions)
        def fitness(pre, epochs):
            global run_count, data_store, fl, fl_store
            run_count += 1
            hparams = create_hparams(pre=pre,  epochs=epochs,
                                     reg_l1=0.05, reg_l2=0.05,
                                     verbose=0)

            mse_avg = 0

            for cnt in range(instance_per_run):
                if plot_dir:
                    plot_name = '{}/{}_{}_run_{}_count_{}'.format(plot_dir, model_mode, loss_mode, run_count, cnt)
                else:
                    plot_name = None
                mse = run_skf(model_mode=model_mode, loss_mode=loss_mode, fl=fl, fl_store=fl_store, hparams=hparams,
                              norm_mask=norm_mask, normalise_labels=normalise_labels, labels_norm=labels_norm,
                              skf_file=hparam_file, skf_sheet='_' + str(run_count) + '_' + str(cnt),
                              k_folds=10, k_shuffle=True,
                              save_model_name='hparam_' + str(run_count) + '_' + str(cnt + 1), save_model=True,
                              save_model_dir=save_model_dir,
                              plot_name=plot_name)
                mse_avg += mse

            mse_avg = mse_avg / instance_per_run
            loss = mse_avg
            end_time = time.time()
            print('**************************************************************************************************\n'
                  'Run Number {} \n'
                  'Instance per run {} \n'
                  'Current run MSE {} \n'
                  'Time Taken: {}\n'
                  '*********************************************************************************************'.format(
                run_count, instance_per_run, mse_avg, end_time - start_time))
            return loss

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=total_run,
                                x0=default_parameters)


    wb = load_workbook(write_dir+'/hparam_results.xlsx')
    hparam_store = np.array(search_result.x_iters)
    results = np.array(search_result.func_vals)
    index = np.arange(total_run)
    toprint = np.concatenate((index.reshape(-1,1),hparam_store, results.reshape(-1, 1)), axis=1)
    if model_mode == 'conv1':
        header = np.array(['index', 'pre', 'filters', 'epochs', 'mse'])
    elif model_mode == 'ann3':
        header = np.array(['index', 'pre', 'epochs', 'mse'])

    toprint = np.concatenate((header.reshape(1,-1), toprint), axis=0)
    sheetname = wb.sheetnames[-1]
    ws = wb[sheetname]
    print_array_to_excel(toprint, (1, 1), ws, axis=2)
    wb.save(write_dir+'/hparam_results.xlsx')
    wb.close()




def grid_hparam_opt(fl, total_run):
    """
     names = ['shared_1_l', 'shared_1_h',
              'shared_2_l', 'shared_2_h',
              'ts_1_l', 'ts_1_h',
              'ts_2_l', 'ts_2_h',
              'epochs_l', 'epochs_h',
              'l1_l', 'l1_h']
    :param model_mode:
    :param loader_file:
    :param total_run:
    :param instance_per_run:
    :param hparam_file:
    :return:
    """

    global run_count, best_loss, data_store, fl_store, best_hparams
    run_count = 0
    best_loss = 100000

    gamma = Real(low=0.1, high=300, name='gamma')
    dimensions = [gamma]
    default_parameters = [30]

    fl_store = fl.create_kf(k_folds=10, shuffle=True)

    @use_named_args(dimensions=dimensions)
    def fitness(gamma):
        global run_count, best_loss, fl_store
        run_count += 1

        # Run k model instance to perform skf
        predicted_labels_store = []
        val_labels = []
        for fold, fl_tuple in enumerate(fl_store):

            (ss_fl, i_ss_fl) = fl_tuple  # ss_fl is training fl, i_ss_fl is validation fl

            model = SVMmodel(fl=ss_fl, gamma=gamma)
            model.train_model(fl=ss_fl)

            # Evaluation
            predicted_labels = model.eval(i_ss_fl)
            predicted_labels_store.extend(predicted_labels)
            val_labels.extend(i_ss_fl.labels)
            del model
            gc.collect()

        # Calculating metrics based on complete validation prediction
        mcc = matthews_corrcoef(y_true=val_labels, y_pred=predicted_labels_store)

        print('**************************************************************************************************\n'
              'Run Number {} \n'
              'Current run MSE {} \n'
              '*********************************************************************************************'.format(
            run_count,  mcc))
        return -mcc

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=total_run,
                                x0=default_parameters)
    plot_convergence(search_result)
    print('Best Loss = {}'.format(search_result.fun))
    print('Best Gamma = {}'.format(search_result.x[0]))



def svr_hparam_opt(fl, total_run):
    """
     names = ['shared_1_l', 'shared_1_h',
              'shared_2_l', 'shared_2_h',
              'ts_1_l', 'ts_1_h',
              'ts_2_l', 'ts_2_h',
              'epochs_l', 'epochs_h',
              'l1_l', 'l1_h']
    :param model_mode:
    :param loader_file:
    :param total_run:
    :param instance_per_run:
    :param hparam_file:
    :return:
    """

    global run_count, best_loss, data_store, fl_store, best_hparams
    run_count = 0
    best_loss = 100000

    gamma = Real(low=0.1, high=300, name='gamma')
    dimensions = [gamma]
    default_parameters = [30]

    fl_store = fl.create_kf(k_folds=10, shuffle=True)

    @use_named_args(dimensions=dimensions)
    def fitness(gamma):
        global run_count, best_loss, fl_store
        run_count += 1

        # Run k model instance to perform skf
        predicted_labels_store = []
        val_labels = []
        for fold, fl_tuple in enumerate(fl_store):

            (ss_fl, i_ss_fl) = fl_tuple  # ss_fl is training fl, i_ss_fl is validation fl

            model = SVRmodel(fl=ss_fl, gamma=gamma)
            model.train_model(fl=ss_fl)

            # Evaluation
            predicted_labels = model.eval(i_ss_fl).flatten().tolist()
            predicted_labels_store.extend(predicted_labels)
            val_labels.extend(i_ss_fl.labels_end.flatten().tolist())
            del model
            gc.collect()

        # Calculating metrics based on complete validation prediction
        mse = mean_squared_error(y_true=val_labels, y_pred=predicted_labels_store)

        print('**************************************************************************************************\n'
              'Run Number {} \n'
              'Current run MSE {} \n'
              '*********************************************************************************************'.format(
            run_count,  mse))
        return mse

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=total_run,
                                x0=default_parameters)
    plot_convergence(search_result)
    print('Best Loss = {}'.format(search_result.fun))
    print('Best Gamma = {}'.format(search_result.x[0]))


def ann_end_hparam_opt(fl_in, total_run, model_selector, write_dir, excel_dir, hparams_excel_dir):
    """
     names = ['shared_1_l', 'shared_1_h',
              'shared_2_l', 'shared_2_h',
              'ts_1_l', 'ts_1_h',
              'ts_2_l', 'ts_2_h',
              'epochs_l', 'epochs_h',
              'l1_l', 'l1_h']
    :param model_mode:
    :param loader_file:
    :param total_run:
    :param instance_per_run:
    :param hparam_file:
    :return:
    """

    global run_count, best_loss, data_store, fl_store, best_hparams
    fl_store = fl_in
    run_count = 0
    best_loss = 100000

    if model_selector == 'svr':
        gamma = Real(low=0.1, high=300, name='gamma')
        dimensions = [gamma]
        default_parameters = [30]
    elif model_selector == 'ann':
        bounds = [[3, 100, ],
                  [3, 100, ],
                  [80, 2500]]
        shared = Integer(low=bounds[0][0], high=bounds[0][1], name='shared')
        shared2 = Integer(low=bounds[1][0], high=bounds[1][1], name='shared2')
        epochs = Integer(low=bounds[2][0], high=bounds[2][1], name='epochs')
        dimensions = [shared, shared2, epochs]
        default_parameters = [5, 5, 100]
    else:
        raise KeyError('model_selector argument is not a valid model')

    if model_selector == 'svr':
        @use_named_args(dimensions=dimensions)
        def fitness(gamma):
            global run_count, fl_store
            run_count += 1
            mse = run_svr(fl_store, write_dir=write_dir, excel_dir=excel_dir,
                          model_selector=model_selector, gamma=gamma)

            print('**************************************************************************************************\n'
                  'Run Number {} \n'
                  'Current run MSE {} \n'
                  '*********************************************************************************************'.format(
                run_count,  mse))
            return mse
    elif model_selector == 'ann':
        @use_named_args(dimensions=dimensions)
        def fitness(shared, shared2, epochs):
            global run_count, fl_store
            run_count += 1
            hparams = create_hparams(shared_layers=[shared, shared2], epochs=epochs,
                                     reg_l1=0.05, reg_l2=0.05,
                                     activation='relu', batch_size=16, verbose=0)
            mse = run_svr(fl_store, write_dir=write_dir, excel_dir=excel_dir,
                          model_selector=model_selector, hparams=hparams, save_name=str(run_count))

            print('**************************************************************************************************\n'
                  'Run Number {} \n'
                  'Current run MSE {} \n'
                  '*********************************************************************************************'.format(
                run_count, mse))
            return mse

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=total_run,
                                x0=default_parameters)
    plot_convergence(search_result)
    print('Best Loss = {}'.format(search_result.fun))
    print('Best Gamma = {}'.format(search_result.x[0]))

    wb = load_workbook(write_dir + '/hparam_results.xlsx')
    hparam_store = np.array(search_result.x_iters)
    results = np.array(search_result.func_vals)
    index = np.arange(total_run)
    toprint = np.concatenate((index.reshape(-1, 1), hparam_store, results.reshape(-1, 1)), axis=1)
    if model_selector == 'svr':
        header = np.array(['index','gamma', 'mse'])
    elif model_selector == 'ann':
        header = np.array(['index', 'shared', 'shared2', 'epochs', 'mse'])
    toprint = np.concatenate((header.reshape(1, -1), toprint), axis=0)
    sheetname = wb.sheetnames[-1]
    ws = wb[sheetname]
    print_array_to_excel(toprint, (1, 1), ws, axis=2)
    wb.save(write_dir + '/hparam_results.xlsx')
    wb.close()
