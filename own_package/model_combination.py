import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

def combine_excel_results(results_excel_dir, end_excel_dir, plot_dir, sheets, fn):
    df = pd.read_excel(end_excel_dir)
    end_store = df.values[:,-2].tolist()
    p_end_store = df.values[:,-1].tolist()

    xls = pd.ExcelFile(results_excel_dir)

    p_y_store_store = []
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        df = df.sort_index()
        p_y_store = np.sinh(df.iloc[:, fn + 20:fn + 39].values)
        p_y_store_store.append(np.concatenate((np.zeros((np.shape(p_y_store)[0], 1)), p_y_store), axis=1).tolist())

    y_store = np.sinh(df.iloc[:, fn + 1:fn + 20].values)
    y_store = np.concatenate((np.zeros((np.shape(y_store)[0], 1)), y_store), axis=1).tolist()

    numel = len(y_store[0])

    p_y_store_store.append(np.sinh(np.mean(np.arcsinh(np.array(p_y_store_store)), axis=0)))
    p_y_store_store = np.array(p_y_store_store)
    combine_mse = np.mean((np.arcsinh(p_y_store_store[-1,:,:])-np.arcsinh(np.array(y_store)))**2)

    for idx, [end, p_end, y] in enumerate(zip(end_store, p_end_store, y_store)):
        x = np.linspace(0, end, num=numel)
        p_x = np.linspace(0, p_end, num=numel)
        temp_p_y_store = p_y_store_store[:, idx, :]

        plt.plot(x, y, c='b', label='Actual Spline Fit')
        plt.scatter(x, y, c='b', marker='+')
        name_store = sheets + ['Combined']

        plt.plot(p_x, temp_p_y_store.tolist()[-1], c='r', label=name_store[-1])
        plt.scatter(p_x, temp_p_y_store.tolist()[-1], c='r', marker='x')

        for p_y, name in zip(temp_p_y_store.tolist()[:-1], name_store[:-1]):
            plt.plot(p_x, p_y, label=name)
            plt.scatter(p_x, p_y,  marker='x')

        plt.legend(loc='upper left')
        plt.title('Expt. ' + str(idx + 1))
        plt.savefig('{}/Expt_{}.png'.format(plot_dir, idx+1),
                    bbox_inches='tight')
        plt.close()

    wb = openpyxl.load_workbook(results_excel_dir)
    ws = wb['Results']
    ws.cell(1,1).value = 'mse'
    ws.cell(1,2).value = combine_mse
    wb.save(results_excel_dir)