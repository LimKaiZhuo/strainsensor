from own_package.svm_classifier import run_classification
from own_package.svr import run_svr
from own_package.others import create_results_directory
from own_package.features_labels_setup import load_data_to_fl
from own_package.hparam_opt import svr_hparam_opt, ann_end_hparam_opt
from own_package.models.models import create_hparams


def selector(case):
    if case == 1:
        write_dir = create_results_directory(results_directory='./results/svm_results with proba')
        run_classification(read_dir='./results/grid full', write_dir=write_dir, gamma=130)
    elif case == 2:
        fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R4.xlsx', normalise_labels=True,
                             norm_mask=[0, 0, 0, 1, 1, 1])
        write_dir = create_results_directory(results_directory='./results/svr_results', excels=['svr_results.xlsx'])
        run_svr(fl=fl, write_dir=write_dir, excel_dir=write_dir + '/svr_results.xlsx',
                model_selector='svr', gamma=0.2694100909858187)
    elif case == 3:
        fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R4.xlsx', normalise_labels=False,
                             norm_mask=[0, 0, 0, 1, 1, 1])
        hparams = create_hparams(shared_layers=[30, 30], epochs=700,
                                 reg_l1=0.05, reg_l2=0.05,
                                 activation='relu', batch_size=16, verbose=0)
        write_dir = create_results_directory(results_directory='./results/svr_results', excels=['svr_results.xlsx'])
        run_svr(fl=fl, write_dir=write_dir, excel_dir=write_dir + '/svr_results.xlsx',
                model_selector='ann', hparams=hparams)
    elif case == 4:
        fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R6_arcsinh.xlsx', normalise_labels=False,
                             norm_mask=[0, 0, 0, 1, 1, 1])
        fl_store = fl.create_kf(k_folds=10, shuffle=True)
        write_dir = create_results_directory(results_directory='./results/end_hparams_results',
                                             excels=['svr_results.xlsx', 'hparam_results.xlsx'])
        ann_end_hparam_opt(fl_store, 150, model_selector='ann',
                           write_dir=write_dir,
                           excel_dir=write_dir + '/svr_results.xlsx',
                           hparams_excel_dir=write_dir + '/hparam_results.xlsx')


selector(1)
