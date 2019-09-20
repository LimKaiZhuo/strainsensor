from own_package.active_learning.acquisition import acquisition_opt
from own_package.spline_analysis import plot_acq_splines
from own_package.others import create_results_directory
from own_package.EXP_acquisition import variance_error_experiement
from own_package.models.models import create_hparams

bounds = [[0, 1],
          [0, 1],
          [200, 2000]]

def selector(run):
    if run == 1:
        write_dir = './results/skf Round 2'

        acquisition_opt(bounds=bounds, model_directory='{}/models'.format(write_dir),
                        svm_directory='./results/svm gamma130/models',
                        loader_file='./excel/Data_loader_spline_full_onehot_R2.xlsx',
                        total_run=3000,
                        batch_runs=5,
                        norm_mask=[0,0,0,1,1,1],
                        acquisition_file='{}/acq.xlsx'.format(write_dir))

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

selector(1)

