from own_package.spline_analysis import plot_predicted_splines, plot_exp_acq_splines, plot_cutoff, plot_arcsinh_predicted_splines

def selector(case):
    if case == 1:
        plot_predicted_splines(write_dir='./results/hparams_opt4', excel_dir='./results/hparams_opt4/skf_results.xlsx',
                               sheets=['conv1_189_0', 'conv1_38_0', 'conv1_22_0'], fn=8)
    elif case == 2:
        plot_exp_acq_splines('./results/acq2', fn=7)
    elif case == 3:
        write_dir = './results/hparams_opt3'
        plot_cutoff(plot_dir='{}/plots'.format(write_dir),
                                       results_excel_dir='{}/skf_results.xlsx'.format(write_dir),
                                       sheets=['ann3_73_0'], fn=6, numel=3)
    elif case == 4:
        write_dir = './results/skf4'
        plot_arcsinh_predicted_splines(plot_dir='{}/plots'.format(write_dir),
                                       results_excel_dir='{}/skf_results.xlsx'.format(write_dir),
                                       end_excel_dir='./results/combine Round 6/end 6e.xlsx',
                                       transformation= 'arcsinh',
                                       sheets=['ann3'], fn=6, numel=99)

selector(4)
