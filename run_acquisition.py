from own_package.active_learning.acquisition import acquisition_opt, l2_points_opt, acquisition_opt_pso_ga
from own_package.spline_analysis import plot_acq_splines
from own_package.others import create_results_directory
from own_package.EXP_acquisition import variance_error_experiement
from own_package.models.models import create_hparams

bounds = [[0, 1],
          [0, 1],
          [200, 2000],
          [0, 2]]


def selector(run, **kwargs):
    if run == 1:
        write_dir = kwargs['write_dir']

        acquisition_opt(bounds=bounds, model_directory='{}/models'.format(write_dir),
                        svm_directory='./results/svm gamma130/models',
                        loader_file='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                        total_run=500,
                        batch_runs=1,
                        normalise_labels=True,
                        norm_mask=[0, 1, 3, 4, 5],
                        acquisition_file='{}/acq.xlsx'.format(write_dir))
    elif run == 1.1:
        write_dir = kwargs['write_dir']
        round = kwargs['round']
        batch = kwargs['batch']
        initial_guess = kwargs['initial_guess']

        params = {'c1': 1.5, 'c2': 1.5, 'wmin': 0.4, 'wmax': 0.9,
                  'ga_iter_min': 2, 'ga_iter_max': 10, 'iter_gamma': 10,
                  'ga_num_min': 5, 'ga_num_max': 20, 'num_beta': 15,
                  'tourn_size': 3, 'cxpd': 0.9, 'mutpd': 0.05, 'indpd': 0.5, 'eta': 0.5,
                  'pso_iter': 15, 'swarm_size': 30}

        acquisition_opt_pso_ga(bounds=bounds, write_dir=write_dir,
                               svm_directory='./results/svm gamma130/models',
                               loader_file='./excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(round),
                               batch_runs=batch, pso_params=params, initial_guess=initial_guess,
                               normalise_labels=False,
                               norm_mask=[0, 1, 3, 4, 5])

    elif run == 2:
        write_dir = './results/actual/conv1 run2'
        plot_acq_splines(write_dir=write_dir, fn=6)

    elif run == 3:
        hparams = create_hparams(shared_layers=[50, 50], ts_layers=[5, 5], cs_layers=[5, 5], epochs=5000, reg_l1=0.05,
                                 reg_l2=0.,
                                 activation='relu', batch_size=16, verbose=0)

        write_dir = create_results_directory('./results/acq',
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['acq_exp'])

        variance_error_experiement('conv1', 'ann', norm_mask=None, labels_norm=False,
                                   loader_file='./excel/Data_loader_spline.xlsx', model_dir=write_dir + '/models/',
                                   hparams=hparams,
                                   results_excel=write_dir + '/acq_exp.xlsx')
    elif run == 4:
        numel = kwargs['numel']
        svm_store = kwargs['svm_store']
        seed_number_expt = kwargs['seed_number_expt']
        total_expt = kwargs['total_expt']
        write_dir = kwargs['write_dir']
        l2_points_opt(numel=numel, write_dir=write_dir, svm_directory=svm_store,
                      seed_number_of_expt=seed_number_expt, total_expt=total_expt)


'''
[[[0.6,0.4,2000,0]],
[[0.595,0.4,1380,0]],
[[0.17,0,1700,0]],
[[0.58,0.4,2000,1]],
[[0.56,0.4,2000,2]]]

[[[0,0.17,2000,0]],
[[0.52,0.39,1980,0]],
[[0.0,0.17,1440,0]],
[[0.96,0.04,200,2]],
[[0.0,0.165,1990,2]]]
'''

# selector(4,numel=210, write_dir='./results/l2 acq', svm_store='./results/svm gamma130/models', seed_number_expt=5, total_expt=30)
selector(1, write_dir='./results/dtr_mse_round_13', round=13, batch=1, initial_guess=[[[0, 0.17, 2000, 0]],
                                                                            [[0.52, 0.39, 1980, 0]],
                                                                            [[0.0, 0.17, 1440, 0]],
                                                                            [[0.96, 0.04, 200, 2]],
                                                                            [[0.0, 0.165, 1990, 2]]])
# selector(1.1, write_dir='./results/dtr_2', round=2, batch=5)
# selector(1.1, write_dir='./results/dtr_3', round=3, batch=8)
# selector(1.1, write_dir='./results/dtr_4', round=4, batch=8)
# selector(1.1, write_dir='./results/dtr_5', round=5, batch=8)
# selector(1.1, write_dir='./results/dtr_6', round=6, batch=8)
