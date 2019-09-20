from own_package.visualization_umap import read_excel_acquisition_data, plot_all_umap

def selector(case):
    if case == 1:
        read_excel_acquisition_data(write_dir='./results/skf9', excel_file='./results/skf9/acq7.xlsx')
    elif case == 2:
        plot_all_umap(read_dir='./results/skf9/acq_fl_data')

selector(2)