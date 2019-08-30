
import pandas as pd
import numpy as np
import gc, pickle
from skopt import gp_minimize, dummy_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from sklearn.metrics import matthews_corrcoef
# Own Scripts
from own_package.models.models import create_hparams
from own_package.svm_classifier import run_classification, SVMmodel
from .cross_validation import run_skf


def hparam_opt(model_mode, loss_mode, norm_mask, labels_norm, loader_file, total_run, instance_per_run=3,
               hparam_file='./excel/hparams_opt.xlsx', plot_dir=None):
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

    global run_count, best_loss, data_store, fl, best_hparams
    run_count = 0
    best_loss = 100000


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

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=total_run,
                                x0=default_parameters)
    plot_convergence(search_result)
    print('Best Loss = {}'.format(search_result.fun))
    print('Best hparams :')
    best_hparams = pd.DataFrame(best_hparams)
    print(best_hparams)


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