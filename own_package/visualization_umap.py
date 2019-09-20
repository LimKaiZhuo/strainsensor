import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from openpyxl import load_workbook
import umap
import pickle, os

def draw_umap(data, scale,  n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=scale)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=scale)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=scale, s=100)
    plt.title(title, fontsize=18)
    plt.savefig(plot_directory + '/Exp ' + str(idx + 1) + ' ' + mode + '.png', bbox_inches='tight')
    plt.close()


def read_excel_acquisition_data(write_dir, excel_file):
    results_directory = write_dir + '/acq_fl_data'
    if os.path.exists(results_directory):
        expand = 1
        while True:
            expand += 1
            new_results_directory = results_directory + str(expand)
            if os.path.exists(new_results_directory):
                continue
            else:
                results_directory = new_results_directory
                break
    os.mkdir(results_directory)
    print('Creating new results directory: {}'.format(results_directory))

    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names[1:]
    print('Loading the following spreadsheets: {} \n From {}'.format(sheet_names, excel_file))
    features_store = []
    labels_store = []
    for sheet in sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet, usecols=[1,2,3,4,5]).values
        features_store.append(df[:, :-1])
        labels_store.append(df[:, -1])

    with open(results_directory + '/features_store', 'wb') as handle:
        pickle.dump(features_store, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(results_directory + '/labels_store', 'wb') as handle:
        pickle.dump(labels_store, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_all_umap(read_dir,  n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    with open(read_dir + '/features_store', 'rb') as handle:
        features_store = pickle.load(handle)
    with open(read_dir + '/labels_store', 'rb') as handle:
        labels_store = pickle.load(handle)
    for idx, store in enumerate(zip(features_store,labels_store)):
        features, labels = store
        draw_umap(data=features, scale=labels,  n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                  metric=metric, title='{}/acq_plot{}'.format(read_dir, idx))
