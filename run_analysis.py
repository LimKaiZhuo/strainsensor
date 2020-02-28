from own_package.analysis import l2_tracker, testset_prediction_results, testset_model_results_to_excel, \
    testset_optimal_combination, save_testset_prediction, eval_combination_on_testset, save_valset_prediction
from own_package.others import create_results_directory


def selector(case, **kwargs):
    if case == 1:
        round_number = kwargs['round_number']
        write_dir = create_results_directory('./results/l2_tracker', excels=['l2_results'])
        l2_tracker(write_excel='{}/l2_results.xlsx'.format(write_dir),
                   final_excel_loader='./excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(round_number),
                   last_idx_store=[11, 16, 21, 29, 37, 45, 69, 77, 85, 93, 101, 109])
    elif case == 2:
        write_dir = create_results_directory('./results/testset_prediction', excels=['testset_prediction'])
        testset_prediction_results(write_excel='{}/testset_prediction.xlsx'.format(write_dir),
                                   model_dir_store=
                                   ['./results/combination {}/models'.format(1),
                                    './results/combination {}/models'.format(2),
                                    './results/combination {}/models'.format(3),
                                    './results/combination {}/models'.format(4),
                                    './results/combination {}/models'.format(5),
                                    './results/combination {}/models'.format(6),
                                    './results/combination {}/models'.format('6e'),
                                    './results/combination {}/models'.format(7),
                                    './results/combination {}/models'.format(8),
                                    './results/combination {}/models'.format(9),
                                    './results/combination {}/models'.format(10),
                                    './results/combination {}/models'.format(11),
                                    './results/combination {}/models'.format(12),
                                    './results/combination {}/models'.format(13)],
                                   excel_loader_dir_store=
                                   ['./excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(1, 1),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(2, 2),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(3, 3),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(4, 4),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(5, 5),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(6, 6),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format('6e', '6e'),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(7, 7),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(8, 8),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(9, 9),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(10, 10),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(11, 11),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(12, 12),
                                    './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13)],
                                   testset_excel_dir='./excel/Data_loader_spline_full_onehot_testset_cut_CM3.xlsx',
                                   rounds=[1, 2, 3, 4, 5, 6, '6e', 7, 8, 9, 10, 11, 12, 13],
                                   fn=6, numel=3)
    elif case == 3:
        write_dir = create_results_directory('./results/testset_prediction', excels=['testset_prediction'])
        testset_prediction_results(write_excel='{}/testset_prediction.xlsx'.format(write_dir),
                                   model_dir_store=
                                   ['./results/combination {}/models'.format('13s3_2')],
                                   excel_loader_dir_store=
                                   ['./excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13)],
                                   testset_excel_dir='./excel/Data_loader_spline_full_onehot_testset_cut_CM3.xlsx',
                                   rounds=['13s'],
                                   fn=6, numel=3)
    elif case == 4:
        write_dir = create_results_directory('./results/testset_all_predictions', excels=['testset_prediction'])
        testset_model_results_to_excel(write_excel='{}/testset_prediction.xlsx'.format(write_dir),
                                       model_dir_store=
                                       ['./results/{}/models'.format('hparams_opt round 13 ANN3 - 2'),
                                        './results/{}/models'.format('hparams_opt round 13 DTR'),
                                        './results/{}/models'.format('hparams_opt round 13 SVR'),
                                        './results/{}/models'.format('hparams_opt round 13 SVR - 2'),
                                        './results/{}/models'.format('hparams_opt round 13 ANN3 - 4'),
                                        './results/{}/models'.format('hparams_opt round 13s ANN3'),
                                        './results/{}/models'.format('hparams_opt round 13s SVR'),
                                        './results/{}/models'.format('hparams_opt round 13s2 ANN3'),
                                        './results/{}/models'.format('hparams_opt round 13s2 SVR'),
                                        './results/{}/models'.format('hparams_opt round 13s2 DTR'),
                                        './results/{}/models'.format('hparams_opt round 13s3 DTR')],
                                       loader_excel=
                                       './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13),
                                       testset_excel_dir='./excel/Data_loader_spline_full_onehot_testset_cut_CM3.xlsx',
                                       fn=6, numel=3, chunks=10)
    elif case == 4.1:
        write_dir = create_results_directory('./results/valtestset_all_predictions', excels=['testset_prediction'])
        testset_model_results_to_excel(write_excel='{}/testset_prediction.xlsx'.format(write_dir),
                                       model_dir_store=
                                       ['./results/10CV/combination 13s/models'],
                                       loader_excel=
                                       './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13),
                                       testset_excel_dir='./excel/Data_loader_spline_full_onehot_testset_cut_CM3.xlsx',
                                       fn=6, numel=3, chunks=10)
    elif case == 4.2:
        write_dir = create_results_directory('./results/combination_13_R13_predictions', excels=['testset_prediction'])
        testset_model_results_to_excel(write_excel='{}/testset_prediction.xlsx'.format(write_dir),
                                       model_dir_store=
                                       ['./results/10CV/combination 13/models'],
                                       loader_excel=
                                       './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13),
                                       testset_excel_dir='./excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13),
                                       fn=6, numel=3, chunks=10)
    elif case == 5:
        hparams = {'n_pop': 1000, 'n_gen': 3000, 'init': [0.5, 0.5]}
        write_dir = create_results_directory('./results/testset_optimal_combination')
        testset_optimal_combination(results_dir=write_dir, y_dat='./results/testset_y.dat',
                                    combination_dat='./results/testset_prediction.dat', hparams=hparams)
    elif case == 5.1:
        hparams = {'n_pop': 1000, 'n_gen': 3000, 'init': [0.5, 0.5]}
        write_dir = create_results_directory('./results/valset_optimal_combination')
        testset_optimal_combination(results_dir=write_dir, y_dat='./results/valset_y.dat',
                                    combination_dat='./results/valset_prediction.dat', hparams=hparams)
    elif case == 6:
        save_testset_prediction(combination_excel='./results/valtestset_all_predictions/testset_prediction.xlsx')
    elif case == 7:
        eval_combination_on_testset(av_excel='./results/testset_optimal_combination_3315/results 3315.xlsx',
                                    y_dat='./results/testset_y.dat',
                                    combination_dat='./results/testset_prediction.dat')
    elif case == 7.1:
        eval_combination_on_testset(av_excel=None,
                                    y_dat='./results/valtestset_y.dat',
                                    combination_dat='./results/valtestset_prediction.dat')
    elif case == 8:
        save_valset_prediction(skf_excel_store=[
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13 ANN3 - 2'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13 DTR'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13 SVR'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13 SVR - 2'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13 ANN3 - 4'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13s ANN3'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13s SVR'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13s2 ANN3'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13s2 SVR'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13s2 DTR'),
            './results/{}/skf_results.xlsx'.format('hparams_opt round 13s3 DTR')])


# selector(1, round_number=11)
selector(4.2)
