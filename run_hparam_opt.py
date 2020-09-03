from own_package.hparam_opt import hparam_opt, hparam_opt_training, hparam_opt_train_val_test, read_hparam_data
from own_package.others import create_results_directory
from own_package.features_labels_setup import load_data_to_fl, load_testset_to_fl
import pickle, os


def selector(case, **kwargs):
    if case == 1:
        loader_excel = kwargs['loader_excel']
        save_model = kwargs['save_model']
        round = kwargs['round']
        smote_numel = kwargs['smote_numel']
        smote_excel = kwargs['smote_excel']
        scoring = kwargs['scoring']
        '''
        write_dir = create_results_directory('./results/hparams_opt round {} DTR'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=True,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        if smote_numel:
            fl_store = fl.fold_smote_kf_augment(numel=smote_numel, k_folds=20, shuffle=True)
        elif smote_excel:
            fl_store = fl.smote_kf_augment(smote_excel=smote_excel, k_folds=20, shuffle=True)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt(model_mode='dtr', loss_mode='dtr', fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                   total_run=120, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)
        '''
        write_dir = create_results_directory('./results/hparams_opt round {} conv1'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=False,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        if smote_numel:
            fl_store = fl.fold_smote_kf_augment(numel=smote_numel, k_folds=10, shuffle=True)
        elif smote_excel:
            fl_store = fl.smote_kf_augment(smote_excel=smote_excel, k_folds=10, shuffle=True)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt(model_mode='conv1', loss_mode='ann', fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                   total_run=50, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)

        '''
        
        write_dir = create_results_directory('./results/hparams_opt round {} ANN3'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=False,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        if smote_numel:
            fl_store = fl.fold_smote_kf_augment(numel=smote_numel, k_folds=20, shuffle=True)
        elif smote_excel:
            fl_store = fl.smote_kf_augment(smote_excel=smote_excel, k_folds=20, shuffle=True)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt(model_mode='ann3', loss_mode='ann', fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                   total_run=120, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)
        '''
    elif case == 2:
        loader_excel = kwargs['loader_excel']
        save_model = kwargs['save_model']
        round = kwargs['round']
        smote_numel = kwargs['smote_numel']
        smote_excel = kwargs['smote_excel']
        scoring = kwargs['scoring']

        write_dir = create_results_directory('./results/hparams_opt round {} DTR'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=False,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        if smote_numel:
            fl_store = fl.fold_smote_kf_augment(numel=smote_numel, k_folds=20, shuffle=True)
        elif smote_excel:
            fl_store = fl.smote_kf_augment(smote_excel=smote_excel, k_folds=20, shuffle=True)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt_training(model_mode='dtr', loss_mode='dtr', fl_in=fl, fl_store_in=fl_store,
                            norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                            total_run=120, instance_per_run=1, write_dir=write_dir,
                            save_model=save_model, save_model_dir=write_dir + '/models/',
                            plot_dir=None)

        '''
        write_dir = create_results_directory('./results/hparams_opt round {} SVR'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=True,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        if smote_numel:
            fl_store = fl.fold_smote_kf_augment(numel=smote_numel, k_folds=20, shuffle=True)
        elif smote_excel:
            fl_store = fl.smote_kf_augment(smote_excel=smote_excel, k_folds=20, shuffle=True)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt(model_mode='svr', loss_mode='svr', fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                   total_run=50, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)

        '''
        '''
        write_dir = create_results_directory('./results/hparams_opt round {} ANN3'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=False,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        if smote_numel:
            fl_store = fl.fold_smote_kf_augment(numel=smote_numel, k_folds=20, shuffle=True)
        elif smote_excel:
            fl_store = fl.smote_kf_augment(smote_excel=smote_excel, k_folds=20, shuffle=True)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt_training(model_mode='ann3', loss_mode='ann', fl_in=fl, fl_store_in=fl_store,
                   norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                   total_run=50, instance_per_run=1, write_dir=write_dir,
                   save_model=save_model, save_model_dir=write_dir + '/models/',
                   plot_dir=None)
       '''
    elif case == 3:
        loader_excel = kwargs['loader_excel']
        save_model = kwargs['save_model']
        round = kwargs['round']
        smote_numel = kwargs['smote_numel']
        smote_excel = kwargs['smote_excel']
        scoring = kwargs['scoring']
        test_excel_dir = kwargs['test_excel_dir']
        write_dir = create_results_directory('./results/hparams_opt round {} DTR'.format(round),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=False,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        test_fl = load_data_to_fl(test_excel_dir, normalise_labels=True, label_type='cutoff',
                                  norm_mask=[0, 1, 3, 4, 5])
        if smote_numel:
            fl_store = fl.fold_smote_kf_augment(numel=smote_numel, k_folds=10, shuffle=True)
        elif smote_excel:
            fl_store = fl.smote_kf_augment(smote_excel=smote_excel, k_folds=10, shuffle=True)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)
        hparam_opt_train_val_test(model_mode='dtr', loss_mode='dtr', fl_in=fl, fl_store_in=fl_store, test_fl=test_fl,
                                  norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                                  total_run=120, instance_per_run=1, write_dir=write_dir,
                                  save_model=save_model, save_model_dir=write_dir + '/models/',
                                  plot_dir=None)
    elif case == 3.1:
        loader_excel = kwargs['loader_excel']
        save_model = kwargs['save_model']
        round = kwargs['round']
        results_name = kwargs['results_name']
        augment_type = kwargs['augment_type']
        smote_numel = kwargs['smote_numel']
        smote_excel = kwargs['smote_excel']
        scoring = kwargs['scoring']
        test_excel_dir = kwargs['test_excel_dir']
        ett_store = kwargs['ett_store']
        model_mode = kwargs['model_mode']
        write_dir = create_results_directory('./results/hparams_opt round {} {}'.format(round, results_name),
                                             folders=['plots', 'models', 'learning rate plots'],
                                             excels=['skf_results', 'hparam_results'])
        fl = load_data_to_fl(loader_excel, normalise_labels=False,
                             label_type='cutoff',
                             norm_mask=[0, 1, 3, 4, 5])
        test_fl = load_testset_to_fl(test_excel_dir, scaler=fl.scaler, norm_mask=[0, 1, 3, 4, 5])
        ett_fl_store = [load_testset_to_fl(x, scaler=fl.scaler, norm_mask=[0, 1, 3, 4, 5]) for x in ett_store]
        if smote_numel:
            if augment_type == 'smote':
                fl_store = fl.fold_smote_kf_augment(numel=smote_numel, k_folds=10, shuffle=True)
            elif augment_type == 'invariant':
                fl_store = fl.fold_invariant_kf_augment(numel=smote_numel, k_folds=10, shuffle=True)
            else:
                raise KeyError('Invalid augment type')
        elif smote_excel:
            fl_store = fl.smote_kf_augment(smote_excel=smote_excel, k_folds=10, shuffle=True)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)

        if model_mode == 'dtr':
            hparam_opt_train_val_test(model_mode='dtr', loss_mode='dtr', fl_in=fl, fl_store_in=fl_store, test_fl=test_fl,
                                      ett_fl_store=ett_fl_store,
                                      norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                                      total_run=40, instance_per_run=1, write_dir=write_dir,
                                      save_model=save_model, save_model_dir=write_dir + '/models/',
                                      plot_dir=None)
        elif model_mode == 'ann':
            hparam_opt_train_val_test(model_mode='ann3', loss_mode='ann', fl_in=fl, fl_store_in=fl_store,
                                      test_fl=test_fl,
                                      ett_fl_store=ett_fl_store,
                                      norm_mask=[0, 1, 3, 4, 5], scoring=scoring,
                                      total_run=20, instance_per_run=1, write_dir=write_dir,
                                      save_model=save_model, save_model_dir=write_dir + '/models/',
                                      plot_dir=write_dir+'/learning rate plots/')
    elif case == 4:
        ett_names = ['I01-1', 'I01-2', 'I01-3',
                     'I05-1', 'I05-2', 'I05-3',
                     'I10-1', 'I10-2', 'I10-3',
                     'I30-1', 'I30-2', 'I30-3',
                     'I50-1', 'I50-2', 'I50-3',
                     '125Test', '125Test I01', '125Test I05','125Test I10']
        write_dir = kwargs['write_dir']
        data_store = []
        for filename in os.listdir(write_dir):
            if filename.endswith(".pkl"):
                with open('{}/{}'.format(write_dir, filename), 'rb') as handle:
                    data_store.extend(pickle.load(handle))
        read_hparam_data(data_store=data_store, write_dir=write_dir, ett_names=ett_names, print_s_df=False,
                         trainset_ett_idx=-4)
        pass


ett_store = ['./excel/ett_30testset_cut Invariant 1.xlsx',
             './excel/ett_30testset_cut Invariant 1 - 2.xlsx',
             './excel/ett_30testset_cut Invariant 1 - 3.xlsx',
             './excel/ett_30testset_cut Invariant 5.xlsx',
             './excel/ett_30testset_cut Invariant 5 - 2.xlsx',
             './excel/ett_30testset_cut Invariant 5 - 3.xlsx',
             './excel/ett_30testset_cut Invariant 10.xlsx',
             './excel/ett_30testset_cut Invariant 10 - 2.xlsx',
             './excel/ett_30testset_cut Invariant 10 - 3.xlsx',
             './excel/ett_30testset_cut Invariant 30.xlsx',
             './excel/ett_30testset_cut Invariant 30 - 2.xlsx',
             './excel/ett_30testset_cut Invariant 30 - 3.xlsx',
             './excel/ett_30testset_cut Invariant 50.xlsx',
             './excel/ett_30testset_cut Invariant 50 - 2.xlsx',
             './excel/ett_30testset_cut Invariant 50 - 3.xlsx',
             './excel/ett_125trainset_cut.xlsx',
             './excel/ett_125trainset_cut Invariant 1.xlsx',
             './excel/ett_125trainset_cut Invariant 5.xlsx',
             './excel/ett_125trainset_cut Invariant 10.xlsx']

#'./excel/ett_125trainset_cut Invariant 30.xlsx'


#selector(case=3.1, round=9, loader_excel='./excel/Data_loader_spline_full_onehot_R9_cut_CM3.xlsx', save_model=False,
#         smote_numel=None, smote_excel=None, scoring='mse', augment_type='invariant',
#         results_name='ann', model_mode='ann',
#         test_excel_dir='./excel/ett_30testset_cut.xlsx',
#         ett_store=ett_store)
#
#selector(case=3.1, round=11, loader_excel='./excel/Data_loader_spline_full_onehot_R11_cut_CM3.xlsx', save_model=False,
#         smote_numel=None, smote_excel=None, scoring='mse', augment_type='invariant',
#         results_name='ann', model_mode='ann',
#         test_excel_dir='./excel/ett_30testset_cut.xlsx',
#         ett_store=ett_store)
#selector(case=3.1, round=12, loader_excel='./excel/Data_loader_spline_full_onehot_R12_cut_CM3.xlsx', save_model=False,
#         smote_numel=None, smote_excel=None, scoring='mse', augment_type='invariant',
#         results_name='ann', model_mode='ann',
#         test_excel_dir='./excel/ett_30testset_cut.xlsx',
#         ett_store=ett_store)
#selector(case=3.1, round=13, loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=False,
#         smote_numel=1100, smote_excel=None, scoring='re', augment_type='smote',
#         results_name='dtr', model_mode='dtr',
#         test_excel_dir='./excel/ett_30testset_cut.xlsx',
#         ett_store=ett_store)
#selector(case=3.1, round=13, loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=False,
#         smote_numel=None, smote_excel=None, scoring='re', augment_type=None,
#         results_name='dtr test', model_mode='dtr',
#         test_excel_dir='./excel/ett_30testset_cut.xlsx',
#         ett_store=ett_store)
#selector(case=3.1, round=12, loader_excel='./excel/Data_loader_spline_full_onehot_R12_cut_CM3.xlsx', save_model=False,
#         smote_numel=None, smote_excel=None, scoring='mse', augment_type='invariant',
#         results_name='ann', model_mode='ann',
#         test_excel_dir='./excel/ett_30testset_cut.xlsx',
#         ett_store=ett_store)
#selector(case=3.1, round=13, loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=False,
#         smote_numel=None, smote_excel=None, scoring='mse', augment_type='invariant',
#         results_name='ann', model_mode='ann',
#         test_excel_dir='./excel/ett_30testset_cut.xlsx',
#         ett_store=ett_store)
for i in [1]:
    selector(case=1, round=i, loader_excel='./excel/Data_loader_spline_full_onehot_R{}.xlsx'.format(i), save_model=True,
             smote_numel=None, smote_excel=None, scoring='mse', augment_type='invariant', model_mode='conv1',
             results_name='conv1_round_{}'.format(i),
             test_excel_dir='./excel/ett_30testset_cut.xlsx',
             ett_store=ett_store)
    #selector(case=3.1, round=i, loader_excel='./excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(i), save_model=True,
    #         smote_numel=100, smote_excel=None, scoring='re', augment_type='invariant', model_mode='dtr',
    #         results_name='dtr_deep_I100_round_{}'.format(i),
    #         test_excel_dir='./excel/ett_30testset_cut.xlsx',
    #         ett_store=ett_store)
#for i in [1,2,3,4,5,6,'6e',7,8,9,10,11,12,13]:
#    selector(case=4, write_dir='./results/hparams_opt round {} DTR_weak_round_{}'.format(i,i))
#selector(case=4, write_dir='./results/hparams_opt round 13 DTR_weak_round_13 - 2')
#selector(case=4, write_dir='./results/hparams_opt round 11 ann')
#selector(case=4, write_dir='./results/hparams_opt round 12 ann')
#selector(case=4, write_dir='./results/hparams_opt round 13 ann')
#selector(case=4, write_dir='./results/hparams_opt round 12 ann')
#selector(case=4, write_dir='./results/hparams_opt round 13 ann')
#selector(case=4, write_dir='./results/hparams_opt round 8 DTR')
#selector(case=4, write_dir='./results/hparams_opt round 9 DTR')
#selector(case=4, write_dir='./results/hparams_opt round 10 DTR')
#selector(case=4, write_dir='./results/hparams_opt round 11 DTR')
#selector(case=4, write_dir='./results/hparams_opt round 12 DTR')
#selector(case=4, write_dir='./results/hparams_opt round 13 DTR')
#selector(case=4, write_dir='./results/hparams_opt round 8 DTR')
#selector(case=4, write_dir='./results/hparams_opt round 13_DTR Invariant 50')
# selector(case=3, round=13, loader_excel='. /excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=True,
#         smote_numel=1100, smote_excel=None, scoring='re',
#         test_excel_dir='./excel/Data_loader_spline_full_onehot_testset_cut_CM3.xlsx')
# selector(case=3, round=13, loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=True,
#         smote_numel=3300, smote_excel=None, scoring='re',
#         test_excel_dir='./excel/Data_loader_spline_full_onehot_testset_cut_CM3.xlsx')

# selector(case=2, round=13, loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=True,
#         smote_numel=None, smote_excel=None, scoring='re')
# selector(case=1, round=1, loader_excel='./excel/Data_loader_spline_full_onehot_R1_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=2, loader_excel='./excel/Data_loader_spline_full_onehot_R2_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=3, loader_excel='./excel/Data_loader_spline_full_onehot_R3_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=4, loader_excel='./excel/Data_loader_spline_full_onehot_R4_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=5, loader_excel='./excel/Data_loader_spline_full_onehot_R5_cut_CM3.xlsx', save_model=True)

# selector(case=1, round=6, loader_excel='./excel/Data_loader_spline_full_onehot_R6_cut_CM3.xlsx', save_model=True)
# selector(case=1, round='6e', loader_excel='./excel/Data_loader_spline_full_onehot_R6e_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=7, loader_excel='./excel/Data_loader_spline_full_onehot_R7_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=8, loader_excel='./excel/Data_loader_spline_full_onehot_R8_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=9, loader_excel='./excel/Data_loader_spline_full_onehot_R9_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=10, loader_excel='./excel/Data_loader_spline_full_onehot_R10_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=11, loader_excel='./excel/Data_loader_spline_full_onehot_R11_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=12, loader_excel='./excel/Data_loader_spline_full_onehot_R12_cut_CM3.xlsx', save_model=True)
# selector(case=1, round=13, loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx', save_model=True)
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
