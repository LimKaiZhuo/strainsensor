from own_package.hparam_opt import hparam_opt
from own_package.others import create_results_directory
from own_package.features_labels_setup import load_data_to_fl

write_dir = create_results_directory('./results/hparams_opt',
                                     folders=['plots', 'models', 'learning rate plots'],
                                     excels=['skf_results', 'hparam_results'])
fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R6_arcsinh.xlsx', normalise_labels=False,
                     norm_mask=[0, 0, 0, 1, 1, 1])
fl_store = fl.create_kf(k_folds=10, shuffle=True)
hparam_opt(model_mode='ann3', loss_mode='p_model', fl_in=fl, fl_store_in=fl_store,
           norm_mask=[0,0,0,1,1,1], normalise_labels=False, labels_norm=False,
           total_run=200, instance_per_run=1, write_dir=write_dir, save_model_dir=write_dir + '/models/',
           plot_dir=None)
'''
write_dir = create_results_directory('./results/hparams_opt',
                                     folders=['plots', 'models', 'learning rate plots'],
                                     excels=['skf_results', 'hparam_results'])
fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R6_arcsinh.xlsx', normalise_labels=False,
                     norm_mask=[0, 0, 0, 1, 1, 1])
fl_store = fl.create_kf(k_folds=10, shuffle=True)
hparam_opt(model_mode='ann3', loss_mode='ann', fl_in=fl, fl_store_in=fl_store,
           norm_mask=[0,0,0,1,1,1], normalise_labels=False, labels_norm=False,
           total_run=200, instance_per_run=1, write_dir=write_dir, save_model_dir=write_dir + '/models/',
           plot_dir=None)

'''