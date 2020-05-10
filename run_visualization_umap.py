from own_package.visualization_umap import read_excel_acquisition_data, plot_all_umap, plot_hparam_rounds, read_hparam_rounds, plot_var, plot_un_hparam_rounds
from own_package.others import create_results_directory

def selector(case):
    if case == 1:
        read_excel_acquisition_data(write_dir='./results/skf9', excel_file='./results/skf9/acq7.xlsx')
    elif case == 2:
        plot_all_umap(read_dir='./results/skf9/acq_fl_data')
    elif case == 3:
        write_dir = create_results_directory('./Plots/rounds')
        excel_store = [['./results/hparams_opt round 1 ANN - 2/overall_summary.xlsx','./results/hparams_opt round 1 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 2 ann - 2/overall_summary.xlsx','./results/hparams_opt round 2 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 3 ann/overall_summary.xlsx','./results/hparams_opt round 3 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 4 ann/overall_summary.xlsx','./results/hparams_opt round 4 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 5 ann/overall_summary.xlsx','./results/hparams_opt round 5 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 6e ann/overall_summary.xlsx','./results/hparams_opt round 6e DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 6 ann/overall_summary.xlsx','./results/hparams_opt round 6 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 7 ann/overall_summary.xlsx','./results/hparams_opt round 7 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 8 ann/overall_summary.xlsx','./results/hparams_opt round 8 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 9 ann/overall_summary.xlsx','./results/hparams_opt round 9 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 10 ann/overall_summary.xlsx','./results/hparams_opt round 10 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 11 ann/overall_summary.xlsx','./results/hparams_opt round 11 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 12 ann/overall_summary.xlsx','./results/hparams_opt round 12 DTR/overall_summary.xlsx'],
                       ['./results/hparams_opt round 13 ann/overall_summary.xlsx','./results/hparams_opt round 13 DTR/overall_summary.xlsx'],]
        rounds = [1,2,3,4,5,6,'6e',7,8,9,10,11,12,13]
        read_hparam_rounds(write_dir=write_dir, excel_store=excel_store, rounds=rounds)
    elif case == 4:
        plot_hparam_rounds(write_dir='./Plots/rounds - 7',
                           metrics=['Train MSE',
                                    'Train MRE',
                                    'Test MSE',
                                    'Test MRE',
                                    'Val MSE',
                                    'Val MRE',
                                    'un125Train MSE',
                                    'un125Train MRE',
                                    ])
    elif case == 5:
        plot_un_hparam_rounds(write_dir='./Plots', excel_dir='./results/active learning round error.xlsx')
    elif case == 6:
        plot_var(excel_dir='./Round 13 GA Combination Summary.xlsx', combi_names=['Round 13','NDA', 'NDA+I', 'NDA+S'])

selector(6)


