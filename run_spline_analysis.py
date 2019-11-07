from own_package.spline_analysis import plot_predicted_splines, plot_exp_acq_splines

def selector(case):
    if case == 1:
        plot_predicted_splines(write_dir='./results/hparams_opt4', excel_dir='./results/hparams_opt4/skf_results.xlsx',
                               sheets=['conv1_189_0', 'conv1_38_0', 'conv1_22_0'], fn=8)
    elif case == 2:
        plot_exp_acq_splines('./results/acq2', fn=7)

selector(1)
