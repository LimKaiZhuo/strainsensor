import keras.backend as K
import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
import openpyxl
from openpyxl import load_workbook
import os, time, gc
from typing import List

from skopt import gp_minimize, gbrt_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

from own_package.features_labels_setup import load_data_to_fl
from own_package.others import print_array_to_excel

from own_package.models.hul_model import HULMultiLossLayer
from own_package.models.models import CrossStitchLayer


def load_model_ensemble(model_directory) -> List:
    """
    Load list of trained keras models from a .h5 saved file that can be used for prediction later
    :param model_directory: model directory where the h5 models are saved in. NOTE: All the models in the directory will
     be loaded. Hence, make sure all the models in there are part of the ensemble and no unwanted models are in the
     directory
    :return: [List: keras models]
    """

    # Loading model names into a list
    model_name_store = []
    directory = model_directory
    for idx, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            model_name_store.append(directory + '/' + filename)
    print('Loading the following models from {}. Total models = {}'.format(directory, len(model_name_store)))

    # Loading model class object into a list
    model_store = []
    for name in model_name_store:
        model_store.append(load_model(name, custom_objects={'CrossStitchLayer': CrossStitchLayer}))
        print('Model {} has been loaded'.format(name))

    return model_store


def model_ensemble_prediction(model_store, features_c_norm):
    """
    Run prediction given one set of feactures_c_norm input, using all the models in model store.
    :param model_store: List of keras models returned by the def load_model_ensemble
    :param features_c_norm: ndarray of shape (1, -1). The columns represents the different features.
    :return: List of metrics.
    """
    predictions_store = []
    for model in model_store:
        predictions = model.predict(features_c_norm)
        predictions = np.concatenate((predictions[0], predictions[1]), axis=1).reshape((-1)).tolist()

        '''
        predictions = model.predict(features_c_norm)
        predictions = [x.item() for x in predictions]
        predictions = np.vstack(predictions).reshape((-1))
        predictions = predictions.tolist()
        '''

        predictions_store.append(predictions)
    predictions_store = np.array(predictions_store)
    predictions_mean = np.mean(predictions_store, axis=0)
    predictions_std = np.std(predictions_store, axis=0)

    return predictions_mean, predictions_std


def features_to_features_input(fl, features_c, features_d) -> np.ndarray:
    """

    :param fl:
    :param features_c:
    :param features_d:
    :return:
    """
    lookup_df_store = fl.lookup_df_store
    # Filling up features descriptors for each discrete feature
    features_dc_store = []
    for item, lookup_df in zip(features_d, lookup_df_store):
        features_dc_store.extend(lookup_df.loc[item, :])

    return np.array(features_c.tolist() + features_dc_store)


def acquisition_opt(bounds, model_directory, norm_mask, loader_file, total_run,
                    acquisition_file='./excel/acquisition_opt.xlsx'):
    """
    bounds = [[5, 200, ],
              [0, 200, ],
              [5, 200, ],
              [0, 200, ],
              [10, 2000],
              [0, 0.3]]
    :param model_mode:
    :param loader_file:
    :param total_run:
    :param instance_per_run:
    :param hparam_file:
    :return:
    """

    model_store = load_model_ensemble(model_directory)

    fl = load_data_to_fl(loader_file, norm_mask=norm_mask)

    space = [Real(low=x[0][0], high=x[0][1], name=x[1]) for x in zip(bounds, fl.features_c_names)] + \
            [Categorical(categories=x[0], name=x[1]) for x in zip(fl.features_d_space, fl.features_d_names)]

    # Choose mass fraction of components to be 0 and the rest maximum.
    x0 = [x[0] if idx in (0, 1) else x[1] for idx, x in enumerate(bounds)] + [x[0] for x in fl.features_d_space]

    prediction_mean_store = []
    prediction_std_store = []
    l2_dist_store = []
    disagreement_store = []
    features_c_count_exclude_d = len(bounds)
    global iter_count
    iter_count = 0

    @use_named_args(space)
    def fitness(**params):
        global iter_count
        iter_count += 1
        if iter_count % 50 == 0:
            print('Current Iteration: {} out of {} '.format(iter_count, total_run))
        # Ensemble Uncertainty
        features = np.array([x for x in params.values()])
        features_c = np.copy(features[:features_c_count_exclude_d])
        features_d = np.copy(features[features_c_count_exclude_d:])
        a_b_sum = np.sum(features_c[0:2])
        if a_b_sum > 1:
            a_score = -10e5 * a_b_sum ** 5
            prediction_mean_store.append([-1] * fl.labels_dim)
            prediction_std_store.append([-1] * fl.labels_dim)
            l2_dist_store.append(-1)
            disagreement_store.append(-1)
        else:
            features_input = features_to_features_input(fl, features_c, features_d)
            features_input_norm = fl.apply_scaling(features_input)
            prediction_mean, prediction_std = model_ensemble_prediction(model_store, features_input_norm)
            # Greedy Sampling
            l2_distance = np.linalg.norm(x=fl.features_c_norm - features_input_norm.reshape((1, -1)), ord=2, axis=1)
            l2_distance = np.min(l2_distance)

            # Overall Acquisition Score. Higher score if l2 distance is larger and uncertainty (std) is larger.
            disagreement = np.sum(prediction_std)
            a_score = l2_distance * disagreement

            # Storing intermediate results into list to print into excel later
            l2_dist_store.append(l2_distance)
            disagreement_store.append(disagreement)
            prediction_mean_store.append(prediction_mean)
            prediction_std_store.append(prediction_std)
        return -a_score

    search_result = forest_minimize(func=fitness,
                                    dimensions=space,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=total_run,
                                    x0=x0,
                                    verbose=False)

    plot_convergence(search_result)
    x_iters = search_result.x_iters
    func_val = -search_result.func_vals
    best_std = -search_result.fun
    best_x = search_result.x

    p_mean_name = np.array(['Pmean_' + str(x) for x in fl.labels_names])
    p_std_name = np.array(['Pstd_' + str(x) for x in fl.labels_names])

    data = np.concatenate((np.array(x_iters),
                           func_val.reshape((-1, 1)),
                           np.array(disagreement_store).reshape((-1, 1)),
                           np.array(l2_dist_store).reshape((-1, 1)),
                           np.array(prediction_mean_store),
                           np.array(prediction_std_store),
                           ), axis=1)

    columns = np.concatenate((np.array(fl.features_c_names[:features_c_count_exclude_d].tolist() + fl.features_d_names),
                              np.array(['A_score']),
                              np.array(['Disagreement']),
                              np.array(['L2']),
                              p_mean_name,
                              p_std_name,
                              ))

    iter_df = pd.DataFrame(data=data,
                           columns=columns)

    iter_df = iter_df.sort_values(by=['A_score'], ascending=False)

    # Checking if skf_file excel exists. If not, create new excel
    if os.path.isfile(acquisition_file) and os.access(acquisition_file,
                                                      os.W_OK):  # Check if file exists and if file is write-able
        print('Writing into' + acquisition_file)
        wb = load_workbook(acquisition_file)
    else:
        # Check if the skf_file name is a proper excel file extension, if not, add .xlsx at the back
        print('File not found. Creating new acquisition file named as : ' + acquisition_file)
        wb = openpyxl.Workbook()
        wb.save(acquisition_file)

    # Creating new worksheet. Even if SNN worksheet already exists, a new SNN1 ws will be created and so on
    wb.create_sheet(title='Acquisition_opt')
    sheet_name = wb.sheetnames[-1]  # Taking the ws name from the back ensures that if SNN1 is the new ws, it works

    # Writing hparam dataframe first
    pd_writer = pd.ExcelWriter(acquisition_file, engine='openpyxl')
    pd_writer.book = wb
    pd_writer.sheets = dict((ws.title, ws) for ws in wb.worksheets)
    iter_df.to_excel(pd_writer, sheet_name)
    pd_writer.save()
    pd_writer.close()
    wb.close()
