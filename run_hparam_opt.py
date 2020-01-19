from own_package.hparam_opt import hparam_opt
from own_package.others import create_results_directory
from own_package.features_labels_setup import load_data_to_fl

def selector(case,**kwargs):
    if case == 1:
        loader_excel = kwargs['loader_excel']
        save_model = kwargs['save_model']
        round = kwargs['round']

        write_dir = create_results_directory('./results/hparams_opt round {} SVR'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=True,
                             label_type='cutoff',
                             norm_mask=[0,1,3,4,5])
        fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt(model_mode='svr', loss_mode='svr', fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0,1,3,4,5],
                   total_run=120, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)

        write_dir = create_results_directory('./results/hparams_opt round {} DTR'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=True,
                             label_type='cutoff',
                             norm_mask=[0,1,3,4,5])
        fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt(model_mode='dtr', loss_mode='dtr', fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0,1,3,4,5],
                   total_run=120, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)

        write_dir = create_results_directory('./results/hparams_opt round {} ANN3'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=False,
                             label_type='cutoff',
                             norm_mask=[0,1,3,4,5])
        fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt(model_mode='ann3', loss_mode='ann', fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0,1,3,4,5],
                   total_run=120, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)
    elif case==2:
        # For classifier
        loader_excel = kwargs['loader_excel']
        save_model = kwargs['save_model']
        round = kwargs['round']

        write_dir = create_results_directory('./results/hparams_opt round {} DTC'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=True,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt(model_mode='dtc', loss_mode=None, fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0, 1, 3, 4, 5],
                   total_run=15, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)

selector(case=2, round=13, loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=True)

#selector(case=1, round=1, loader_excel='./excel/Data_loader_spline_full_onehot_R1_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=2, loader_excel='./excel/Data_loader_spline_full_onehot_R2_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=3, loader_excel='./excel/Data_loader_spline_full_onehot_R3_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=4, loader_excel='./excel/Data_loader_spline_full_onehot_R4_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=5, loader_excel='./excel/Data_loader_spline_full_onehot_R5_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=6, loader_excel='./excel/Data_loader_spline_full_onehot_R6_cut_CM3.xlsx', save_model=True)
#selector(case=1, round='6e', loader_excel='./excel/Data_loader_spline_full_onehot_R6e_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=7, loader_excel='./excel/Data_loader_spline_full_onehot_R7_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=8, loader_excel='./excel/Data_loader_spline_full_onehot_R8_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=9, loader_excel='./excel/Data_loader_spline_full_onehot_R9_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=10, loader_excel='./excel/Data_loader_spline_full_onehot_R10_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=11, loader_excel='./excel/Data_loader_spline_full_onehot_R11_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=12, loader_excel='./excel/Data_loader_spline_full_onehot_R12_cut_CM3.xlsx', save_model=True)
#selector(case=1, round=13, loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=True)
'''
write_dir = create_results_directory('./results/hparams_opt',
                                     folders=['plots', 'models', 'learning rate plots'],
                                     excels=['skf_results', 'hparam_results'])
fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R6_arcsinh.xlsx', normalise_labels=False,
                     norm_mask=[0,1,3,4,5])
fl_store = fl.create_kf(k_folds=10, shuffle=True)
hparam_opt(model_mode='ann3', loss_mode='ann', fl_in=fl, fl_store_in=fl_store,
           norm_mask=[0,1,3,4,5], normalise_labels=False, labels_norm=False,
           total_run=200, instance_per_run=1, write_dir=write_dir, save_model_dir=write_dir + '/models/',
           plot_dir=None)

'''