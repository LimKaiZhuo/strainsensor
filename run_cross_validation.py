from own_package.cross_validation import run_skf, run_skf_with_training_error, run_skf_train_val_test_error, run_eval_model_on_train_val_test_error
from own_package.models.models import create_hparams
from own_package.others import create_results_directory, create_excel_file
from own_package.spline_analysis import plot_arcsinh_predicted_splines, plot_cutoff
from own_package.features_labels_setup import load_data_to_fl, load_testset_to_fl
from own_package.analysis import testset_model_results_to_excel
from own_package.hparam_opt import read_hparam_data
import openpyxl, os, pickle
from own_package.active_learning.acquisition import load_model_ensemble

def run_skf_conv1(inputs, plot_spline, smote_numel):
    shared, end, pre, filters, epochs, label_type = inputs
    hparams = create_hparams(shared_layers=[30, 30], ts_layers=[5, 5], cs_layers=[5, 5],
                             shared=shared, end=end, pre=pre, filters=filters, epochs=epochs,
                             reg_l1=0.05, reg_l2=0.,
                             max_depth=5, num_est=200,
                             activation='relu', batch_size=16, verbose=0)

    write_dir = create_results_directory('./results/skf',
                                         folders=['plots', 'models', 'learning rate plots'],
                                         excels=['skf_results'])
    fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                         label_type=label_type,
                         normalise_labels=True,
                         norm_mask=[0, 0, 0, 1, 1, 1])

    if smote_numel:
        fl_store = fl.fold_smote_kf_augment(k_folds=10, shuffle=True, numel=smote_numel)
    else:
        fl_store = fl.create_kf(k_folds=10, shuffle=True)

    run_skf(model_mode='dtr', loss_mode='dtr', fl=fl, fl_store=fl_store, hparams=hparams,
            skf_file=write_dir + '/skf_results.xlsx',
            skf_sheet=None,
            k_folds=10, k_shuffle=True, save_model=True, save_model_name=None, save_model_dir=write_dir + '/models/',
            plot_name=write_dir + '/learning rate plots/plot')
    if plot_spline:
        if label_type == 'points':
            plot_arcsinh_predicted_splines(plot_dir='{}/plots'.format(write_dir),
                                           results_excel_dir='{}/skf_results.xlsx'.format(write_dir),
                                           end_excel_dir='./results/combine Round 6/end 6e.xlsx',
                                           sheets=['ann3'], fn=6, numel=100)
        elif label_type == 'cutoff':
            plot_cutoff(plot_dir='{}/plots'.format(write_dir),
                        results_excel_dir='{}/skf_results.xlsx'.format(write_dir),
                        sheets=['ann3'], fn=6, numel=3)

    write_excel = create_excel_file('{}/training_error.xlsx'.format(write_dir))
    testset_model_results_to_excel(write_excel=write_excel, model_dir_store=['{}/models'.format(write_dir)],
                                   loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                                   testset_excel_dir='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                                   fn=6, numel=3, chunks=10)

    return write_dir


def run_skf_with_te(inputs_store, loader_excel, smote_numel, mode, name, learningrate=0.001, eval_model_dir=None):
    write_dir = create_results_directory('./results/{}'.format(name),
                                         folders=['plots', 'models', 'learning rate plots'],
                                         excels=['skf_results', 'te.xlsx'])
    data_store = []
    loss = 'mse'
    if eval_model_dir:
        inputs_store = load_model_ensemble(eval_model_dir)

    for inputs in inputs_store:
        fl = load_data_to_fl(loader_excel,
                             label_type='cutoff',
                             normalise_labels=False,
                             norm_mask=[0, 1, 3, 4, 5])

        test_excel_dir = './excel/ett_30testset_cut.xlsx'
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

        test_fl = load_testset_to_fl(test_excel_dir, scaler=fl.scaler, norm_mask=[0, 1, 3, 4, 5])
        ett_fl_store = [load_testset_to_fl(x, scaler=fl.scaler, norm_mask=[0, 1, 3, 4, 5]) for x in ett_store]

        if smote_numel:
            fl_store = fl.fold_smote_kf_augment(k_folds=10, shuffle=True, numel=smote_numel)
        else:
            fl_store = fl.create_kf(k_folds=10, shuffle=True)

        if eval_model_dir:
            val_score, train_score, data = run_eval_model_on_train_val_test_error(fl=fl,
                                                                        fl_store=fl_store, test_fl=test_fl,
                                                                        ett_fl_store=ett_fl_store,
                                                                        model_name='hparams_opt_makeup',model=inputs,)
        else:
            pre, epochs = inputs
            hparams = create_hparams(shared_layers=[30, 30], ts_layers=[5, 5], cs_layers=[5, 5],
                                     learning_rate=learningrate,
                                     shared=0, end=0, pre=pre, filters=0, epochs=epochs,
                                     reg_l1=0.0005, reg_l2=0, loss=loss,
                                     max_depth=pre, num_est=epochs,
                                     epsilon=0.0001, c=0.001,
                                     activation='relu', batch_size=16, verbose=0)

            if mode == 'ann':
                model_mode = 'ann3'
                loss_mode = 'ann'
            elif mode=='dtr':
                model_mode = 'dtr'
                loss_mode = 'dtr'


            val_score, train_score, data = run_skf_train_val_test_error(model_mode=model_mode, loss_mode=loss_mode,
                                                                        fl=fl,
                                                                        fl_store=fl_store, test_fl=test_fl,
                                                                        ett_fl_store=ett_fl_store,
                                                                        model_name='{}_{}_{}_{}'.format(write_dir,
                                                                                                     model_mode, pre, epochs),
                                                                        hparams=hparams,
                                                                        k_folds=10, scoring='mse',
                                                                        save_model_name='/{}_{}_{}'.format(mode, pre, epochs),
                                                                        save_model=True,
                                                                        save_model_dir=write_dir + '/models',
                                                                        plot_name='{}/{}'.format(write_dir, str(inputs)))
        ett_names = ['I01-1', 'I01-2', 'I01-3',
                     'I05-1', 'I05-2', 'I05-3',
                     'I10-1', 'I10-2', 'I10-3',
                     'I30-1', 'I30-2', 'I30-3',
                     'I50-1', 'I50-2', 'I50-3',
                     '125Test', '125Test I01', '125Test I05', '125Test I10']
        if eval_model_dir:
            data.append([1, 1])
        else:
            data.append([pre,epochs])
        data_store.append(data)
        with open('{}/data_store.pkl'.format(write_dir), "wb") as file:
            pickle.dump(data_store, file)
    read_hparam_data(data_store=data_store, write_dir=write_dir, ett_names=ett_names, print_s_df=False,
                     trainset_ett_idx=-4)




def run_skf_with_te_nofolds(inputs, plot_spline, smote_numel):
    shared, end, pre, filters, epochs, label_type = inputs
    hparams = create_hparams(shared_layers=[30, 30], ts_layers=[5, 5], cs_layers=[5, 5],
                             shared=shared, end=end, pre=pre, filters=filters, epochs=epochs,
                             reg_l1=0.0005, reg_l2=0.,
                             max_depth=100, num_est=1000,
                             epsilon=0.0001, c=0.001,
                             activation='relu', batch_size=4, verbose=0)

    write_dir = create_results_directory('./results/skf',
                                         folders=['plots', 'models', 'learning rate plots'],
                                         excels=['skf_results', 'te.xlsx'])
    fl = load_data_to_fl('./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                         label_type=label_type,
                         normalise_labels=False,
                         norm_mask=[0, 1, 3, 4, 5])

    if smote_numel:
        fl_store = fl.fold_smote_kf_augment(k_folds=10, shuffle=True, numel=smote_numel)
    else:
        fl_store = fl.create_kf(k_folds=10, shuffle=True)

    run_skf_with_training_error(model_mode='ann3', loss_mode='ann', fl=fl, fl_store=[[fl,fl]], hparams=hparams,
                                skf_file=write_dir + '/skf_results.xlsx',
                                te_sheet=write_dir + '/te.xlsx',
                                skf_sheet=None,
                                k_folds=10, k_shuffle=True, save_model=True, save_model_name=None,
                                save_model_dir=write_dir + '/models/',
                                plot_name=write_dir + '/learning rate plots/plot')

    write_excel = create_excel_file('{}/training_error.xlsx'.format(write_dir))
    testset_model_results_to_excel(write_excel=write_excel, model_dir_store=['{}/models'.format(write_dir)],
                                   loader_excel='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                                   testset_excel_dir='./excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                                   fn=6, numel=3, chunks=10)

#hparams_store = [[1000, 4000], [995, 3986], [1000, 4000], [1000, 4000], [1000, 4000]]
#for hparam in hparams_store:
#    inputs = [0, 0, hparam[0], 0, hparam[1], 'cutoff']
#    write_dir = run_skf_with_te_nofolds(inputs, plot_spline=False, smote_numel=1100)
#write_dir = run_skf_conv1(inputs, plot_spline=False, smote_numel=1050)
round_store = [4,5,6,'6e',7,8,9,10,11,12,13]
hparams_store = [[[235, 285], [374, 289], [427, 275]],
                 [[341, 27], [340, 29], [336, 25]],
                 [[179, 76], [491, 73], [111, 169]],
                 [[227, 24], [231, 152], [228, 23]],
                 [[494, 87], [420, 347], [322, 402]],
                 [[107, 4], [401, 194], [282, 294]],
                 [[455, 115], [468, 235], [446, 470]],
                 [[139, 11], [334, 70], [192, 188]],
                 [[16, 407], [208, 90], [243, 60]],
                 [[412, 90], [273, 200], [82, 26]],
                 [[3, 31], [462, 99], [332, 14]],
                 [[3, 44], [3, 20], [3, 68]],
                 [[486, 432], [487, 174], [489, 249]],
                 [[235, 217], [302, 166], [424, 32]]]
loader_excel_store = [
                      './excel/Data_loader_spline_full_onehot_R4_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R5_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R6_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R6e_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R7_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R8_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R9_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R10_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R11_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R12_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',]
'''
'./excel/Data_loader_spline_full_onehot_R1_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R2_cut_CM3.xlsx',
                      './excel/Data_loader_spline_full_onehot_R3_cut_CM3.xlsx',
[[1000, 4000], [995, 3986]],
                 [[1000, 3446], [1000, 3964], [1000, 3985]],
                 [[39, 3223], [118, 2595], [128, 1861]],
'''
#hparams_store = [
#                 [[1000, 400], [987,396]],
#                 [[651, 3657], [278, 3564], [426, 3468]],
#                 [[143, 3177], [120, 2988], [126, 3106]],
#                 [[1000, 1067], [1000, 1017], [499, 2779]],
#                 [[782, 2190], [764, 2206], [730, 2133]],
#                 [[40, 7824], [125, 4653], [55, 5554]],
#                 [[286, 6011], [294, 5795], [364, 4187]],
#                 [[140, 7960], [142, 7671], [158, 5904]],
#                 [[871, 2492], [593, 3548], [670, 2283]],
#                 [[302, 7393], [259, 7412], [306, 7447]],
#                 [[485, 3477], [474, 1791], [138, 5169]]]
#hparams_store = [[[0.001, 300, 500, 'haitao'],[0.001, 600, 500, 'haitao']]]
hparams_store = [
                 [[485, 3477]],]
round_store = [13]
#for round, hparam, loader_excel in zip(round_store, hparams_store, loader_excel_store):
    #run_skf_with_te(hparam, './excel/Data_loader_spline_full_onehot_R13test_cut_CM3.xlsx',
    #                smote_numel=None, mode='ann', name='{}_{}'.format('anntest_mse',round), loss='mse')
    #run_skf_with_te(hparam, './excel/Data_loader_spline_full_onehot_R13test_cut_CM3.xlsx',
    #                smote_numel=None, mode='ann', name='{}_{}'.format('anntest_mape', round), loss='mape')
    #run_skf_with_te(hparam, './excel/Data_loader_spline_full_onehot_R13test_cut_CM3.xlsx',
    #                smote_numel=None, mode='ann', name='{}_{}'.format('anntest_haitao', round), loss='haitao')
    #run_skf_with_te(hparam, './excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
    #                smote_numel=None, mode='ann', name='{}_{}'.format('ann_mse',round), loss='mse')
    #run_skf_with_te(hparam, './excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
    #                smote_numel=None, mode='ann', name='{}_{}'.format('ann_mape', round), loss='mape')
    #run_skf_with_te(hparam, loader_excel,
    #                smote_numel=None, mode='ann', name='{}_{}'.format('ann_mse_round_test', round), learningrate=0.001/2)
    #run_skf_with_te([[378,1], [114,1], [124,98]], './excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
    #                smote_numel=None, mode='dtr', name='{}_{}'.format('dtr_makeup_round', round),)
# plot_predicted_splines(write_dir='./results/skf10 archsinh', excel_dir='./results/skf10 archsinh/skf_results.xlsx', sheets=['conv1'], fn=6)
# './excel/smote_1.xlsx'
run_skf_with_te([1], './excel/Data_loader_spline_full_onehot_R13_cut_CM3.xlsx',
                eval_model_dir='./results/inverse_design/models',
                smote_numel=None, mode='dtr', name='{}'.format('dtr_makeup_round'),)


