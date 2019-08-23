import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_predicted_splines(write_dir, fn):
    read_excel_file = '{}/skf_results.xlsx'.format(write_dir)

    df = pd.read_excel(read_excel_file, sheet_name=1)
    df = df.sort_index()

    end_store = df.iloc[:, fn + 1].values
    p_end_store = df.iloc[:, fn + 22].values

    y_store = df.iloc[:, fn + 2:fn + 22].values.tolist()
    p_y_store = df.iloc[:, fn + 23:fn + 43].values.tolist()

    numel = len(y_store[0])

    for idx, [end, p_end, y, p_y] in enumerate(zip(end_store, p_end_store, y_store, p_y_store)):
        x = np.linspace(0, end, num=numel)
        p_x = np.linspace(0, p_end, num=numel)

        plt.plot(x, y, c='b', label='Actual Spline Fit')
        plt.plot(p_x, p_y, c='r', label='Predicted Spline Fit')
        plt.scatter(x, y, c='b', marker='+')
        plt.scatter(p_x, p_y, c='r', marker='x')
        plt.legend(loc='upper left')
        plt.title('Expt. ' + str(idx + 1))
        plt.savefig(write_dir + '/plots' + '/Expt ' + str(idx + 1) + ' _spline' + '.png', bbox_inches='tight')
        plt.close()


def plot_exp_acq_splines(write_dir, fn):
    read_excel_file = '{}/acq_exp.xlsx'.format(write_dir)

    df = pd.read_excel(read_excel_file, sheet_name=-1)
    df = df.sort_index()

    end_store = df.iloc[:, fn].values
    p_end_store = df.iloc[:, fn + 21].values

    y_store = df.iloc[:, fn + 1:fn + 21].values.tolist()
    p_y_store = df.iloc[:, fn + 22:fn + 42].values.tolist()

    y_std_store = df.iloc[:, fn + 43:fn + 63].values
    e_std_store = df.iloc[:, fn + 42].values
    total_std_store = df.iloc[:, -1].values
    numel = len(y_store[0])

    for idx, [end, p_end, y, p_y, y_std, e_std, total_std] in enumerate(zip(end_store, p_end_store, y_store,
                                                                            p_y_store, y_std_store, e_std_store,
                                                                            total_std_store)):
        x = np.linspace(0, end, num=numel)
        p_x = np.linspace(0, p_end, num=numel)

        plt.plot(x, y, c='b', label='Actual Spline Fit')
        plt.errorbar(p_x, p_y, yerr=y_std, c='r', label='Predicted Spline Fit')
        plt.scatter(x, y, c='b', marker='+')
        plt.scatter(p_x, p_y, c='r', marker='x')
        plt.plot([], [], ' ', label='End Pt. std: {}'.format(e_std))
        plt.plot([],[],' ', label='Total std: {}'.format(total_std))
        plt.legend(loc='upper left')
        plt.title('Expt. ' + str(idx + 1))
        plt.savefig(write_dir + '/plots' + '/Expt ' + str(idx + 1) + ' _spline' + '.png', bbox_inches='tight')
        plt.close()


def plot_acq_splines(write_dir, fn):
    read_excel_file = '{}/acq.xlsx'.format(write_dir)

    df = pd.read_excel(read_excel_file, sheet_name=-1)

    p_end_store = df.iloc[:, fn + 3].values

    p_y_store = df.iloc[:, fn + 4:fn + 24].values.tolist()

    numel = len(p_y_store[0])

    for idx, [p_end, p_y] in enumerate(zip(p_end_store, p_y_store)):
        p_x = np.linspace(0, p_end, num=numel)

        plt.plot(p_x, p_y, c='b', label='Predicted Spline Fit')
        plt.scatter(p_x, p_y, c='r', marker='x')
        plt.legend(loc='upper left')
        plt.title('Expt. ' + str(idx + 1))
        plt.savefig(write_dir + '/acq_plots' + '/Expt ' + str(idx + 1) + ' _spline' + '.png', bbox_inches='tight')
        plt.close()


def plot_predicted_poly(write_dir):
    read_excel_file = '{}/skf_results.xlsx'.format(write_dir)

    df = pd.read_excel(read_excel_file, sheet_name='ann')
    df = df.sort_index()

    end_store = df.iloc[:, 8].values
    p_end_store = df.iloc[:, 29].values

    y_store = df.iloc[:, 9:29].values.tolist()
    p_y_store = df.iloc[:, 30:50].values.tolist()

    numel = len(y_store[0])

    for idx, [end, p_end, y, p_y] in enumerate(zip(end_store, p_end_store, y_store, p_y_store)):
        x = np.linspace(0, end, num=numel)
        p_x = np.linspace(0, p_end, num=numel)

        plt.plot(x, y, c='b', label='Actual Spline Fit')
        plt.plot(p_x, p_y, c='r', label='Predicted Spline Fit')
        plt.scatter(x, y, c='b', marker='+')
        plt.scatter(p_x, p_y, c='r', marker='x')
        plt.legend(loc='upper left')
        plt.title('Expt. ' + str(idx + 1))
        plt.savefig(write_dir + '/plots' + '/Expt ' + str(idx + 1) + ' _spline' + '.png', bbox_inches='tight')
        plt.close()
