from own_package.spline_analysis import plot_predicted_splines, plot_exp_acq_splines

def selector(case):
    if case == 1:
        plot_predicted_splines('./results/skf', fn=8)
    elif case == 2:
        plot_exp_acq_splines('./results/acq2', fn=7)

selector(1)
