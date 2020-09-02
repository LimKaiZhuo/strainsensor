from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle, time
import uncertainpy as un
import openpyxl

from own_package.active_learning.acquisition import model_ensemble_prediction, load_model_ensemble
from own_package.features_labels_setup import load_data_to_fl


def features_gsa(write_dir, model_dir, fl_loader_file, N):
    problem = {
        'num_vars': 4,
        'names': ['','','',''],
        'bounds': [[0,1],[0,1],[200,2000], [0,1]],
    }
    param_values = saltelli.sample(problem, N, calc_second_order=True)

    param_values = param_values[param_values[:,0]+param_values[:,1]<=1]

    features_c = param_values[:,:-1]
    onehot = param_values[:,-1]
    onehot_store = []
    for single in onehot:
        if single <=1/3:
            onehot_store.append([1,0,0])
        elif single<=2/3:
            onehot_store.append([0,1,0])
        else:
            onehot_store.append([0,0,1])
    features = np.concatenate((features_c, np.array(onehot_store)), axis=1)

    fl = load_data_to_fl(fl_loader_file, norm_mask=[0, 1, 3, 4, 5], normalise_labels=False, label_type='cutoff')
    features_input_norm = fl.apply_scaling(features)

    model_store = load_model_ensemble(model_dir)
    outputs = model_ensemble_prediction(model_store, features_input_norm)

    si_output = sobol.analyze(problem, outputs, calc_second_order=True, print_to_console=True)

    print('pass')


features_gsa(None,None,100)


