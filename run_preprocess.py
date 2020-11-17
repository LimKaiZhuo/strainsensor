from own_package.preprocess import read_excel_data, read_excel_data_to_spline, read_grid_data
from own_package.others import create_results_directory



def run_preprocess(select):
    if select == 1:
        write_dir = create_results_directory('./results/preprocess_poly', excels=['results'])

        read_excel_data('./excel/Raw_Data_caa_090219.xlsx', write_excel_file=write_dir + '/results.xlsx',
                        normalise_r=False, mode='multipoly_cutoff', plot_directory=write_dir + '/plots', poly=2)
    elif select == 2:
        write_dir = create_results_directory('./results/preprocess')
        read_excel_data_to_spline(read_excel_file='./excel/Raw_Data_Round2_removed_outlier_a.xlsx',
                                  write_dir=write_dir, discrete_points=20, spline_selector=1)

    elif select == 3:
        write_dir = create_results_directory('./results/grid')
        read_grid_data(read_excel_file='./excel/grid.xlsx', write_dir=write_dir)

run_preprocess(2)

