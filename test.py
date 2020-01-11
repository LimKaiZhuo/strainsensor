import numpy as np
import pickle
import matplotlib.pyplot as plt
from own_package.active_learning.acquisition import features_to_features_input, \
    svm_ensemble_prediction, load_svm_ensemble, load_model_ensemble, model_ensemble_prediction
from own_package.svm_classifier import SVMmodel
from own_package.hparam_opt import grid_hparam_opt
from own_package.spline_analysis import plot_arcsinh_predicted_splines
from own_package.model_combination import combine_excel_results, cutoff_combine_excel_results, mse_tracker, final_prediction_results
from own_package.others import create_results_directory


def test(selector, number=None):
    if selector == 1:
        svm_store = load_svm_ensemble('./results/svm_results/models')
        x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        composition = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        prediction, distance = svm_ensemble_prediction(svm_store, composition)
        plt.scatter(composition[:, 0], composition[:, 1], c=distance)
        plt.colorbar()
        plt.savefig('./results/distance map.png', bbox_inches='tight')
        plt.close()
        plt.scatter(composition[:, 0], composition[:, 1], c=prediction)
        plt.colorbar()
        plt.savefig('./results/prediction map.png', bbox_inches='tight')
        plt.close()
        with open('results/grid full/grid_data', 'rb') as handle:
            fl = pickle.load(handle)
        plt.scatter(fl.features[:, 0], fl.features[:, 1], c=fl.labels)
        plt.colorbar()
        plt.savefig('./results/actual map.png', bbox_inches='tight')
        plt.close()

        model = SVMmodel(fl=fl, gamma=130)
        model.train_model(fl=fl)
        prediction, distance = svm_ensemble_prediction([model], composition)
        plt.scatter(composition[:, 0], composition[:, 1], c=distance)
        plt.colorbar()
        plt.savefig('./results/distance map2.png', bbox_inches='tight')
        plt.close()
        plt.scatter(composition[:, 0], composition[:, 1], c=prediction)
        plt.colorbar()
        plt.savefig('./results/prediction map2.png', bbox_inches='tight')
        plt.close()

    elif selector == 2:
        with open('results/grid full/grid_data', 'rb') as handle:
            fl = pickle.load(handle)

        grid_hparam_opt(fl, 300)
    elif selector == 3:
        composition = np.array([0.175763935003216, 0.195036471863385])
        svm_store = load_svm_ensemble('./results/svm gamma130/models')
        prediction, distance = svm_ensemble_prediction(svm_store, composition)
        print('prediction: {}\ndistance: {}'.format(prediction, distance))
    elif selector == 4:
        write_dir = './results/skf3'
        plot_arcsinh_predicted_splines(plot_dir='{}/plots'.format(write_dir),
                                       results_excel_dir='{}/skf_results.xlsx'.format(write_dir),
                                       end_excel_dir='./results/combine Round 6/end 6.xlsx',
                                       transformation='arcsinh',
                                       sheets=['ann3'], fn=6, numel=99)
    elif selector == 5:
        combine_excel_results(results_excel_dir='./results/combine Round 6/combination.xlsx',
                              end_excel_dir='./results/combine Round 6/end 6.xlsx',
                              plot_dir='./results/combine Round 6/plots',
                              sheets=['ann3_115_0', 'ann3_190_0 sqrt', 'conv1_40_0', 'conv1_158_0 sqrt'],
                              fn=6)
    elif selector == 6:
        cutoff_combine_excel_results(dir_store=['./results/hparams_opt Round {} SVR'.format(number),
                                                './results/hparams_opt Round {} DTR'.format(number),
                                                './results/hparams_opt Round {} ANN3'.format(number)],
                                     sheets=['svr', 'dtr', 'ann3'],
                                     results_excel_dir='./results/combination {}/combination CM R{}.xlsx'.format(number, number),
                                     plot_dir='./results/combination {}/plots'.format(number),
                                     plot_mode=False,
                                     fn=6, numel=3)
    elif selector == 6.1:
        cutoff_combine_excel_results(dir_store=['./results/hparams_opt Round {} DTR'.format(number),
                                                './results/hparams_opt Round {} ANN3 - 2'.format(number)],
                                     sheets=['dtr', 'ann3'],
                                     results_excel_dir='./results/combination {}/combination CM R{}.xlsx'.format(number, number),
                                     plot_dir='./results/combination {}/plots'.format(number),
                                     plot_mode=False,
                                     fn=6, numel=3)

    elif selector == 7:
        model_store = load_model_ensemble('./results/skf13/models')
        mean, std = model_ensemble_prediction(model_store, np.array([[0.5, 0.5, 0.5, 0, 1, 0]]))
        print(mean, std)

    elif selector == 8:
        mse_tracker(excel_store=['./results/combination {}/combination CM R{}.xlsx'.format(1, 1),
                                 './results/combination {}/combination CM R{}.xlsx'.format(2, 2),
                                 './results/combination {}/combination CM R{}.xlsx'.format(3, 3),
                                 './results/combination {}/combination CM R{}.xlsx'.format(4, 4),
                                 './results/combination {}/combination CM R{}.xlsx'.format(5, 5),
                                 './results/combination {}/combination CM R{}.xlsx'.format(6, 6),
                                 './results/combination {}/combination CM R{}.xlsx'.format('6e', '6e'),
                                 './results/combination {}/combination CM R{}.xlsx'.format(7, 7),
                                 './results/combination {}/combination CM R{}.xlsx'.format(8, 8),
                                 './results/combination {}/combination CM R{}.xlsx'.format(9,9),
                                 './results/combination {}/combination CM R{}.xlsx'.format(10,10),
                                 './results/combination {}/combination CM R{}.xlsx'.format(11,11),
                                 './results/combination {}/combination CM R{}.xlsx'.format(12,12),
                                 './results/combination {}/combination CM R{}.xlsx'.format(13,13)],
                    write_excel='./MSE tracker.xlsx',
                    rounds=[1,2,3,4,5,6,'6e',7,8,9, 10, 11,12,13],
                    headers=['SVR', 'DTR', 'ANN3', 'Combined'],
                    fn=6, numel=3)
    elif selector == 9:
        write_dir = create_results_directory(results_directory='./results/final_prediction', excels=['final_prediction'])
        final_prediction_results(write_excel='{}/final_prediction.xlsx'.format(write_dir),
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
                                  './results/combination {}/models'.format(13)]
                                 ,
                                 combined_excel_store=
                                 ['./results/combination {}/combination CM R{}.xlsx'.format(1, 1),
                                  './results/combination {}/combination CM R{}.xlsx'.format(2, 2),
                                  './results/combination {}/combination CM R{}.xlsx'.format(3, 3),
                                  './results/combination {}/combination CM R{}.xlsx'.format(4, 4),
                                  './results/combination {}/combination CM R{}.xlsx'.format(5, 5),
                                  './results/combination {}/combination CM R{}.xlsx'.format(6, 6),
                                  './results/combination {}/combination CM R{}.xlsx'.format('6e', '6e'),
                                  './results/combination {}/combination CM R{}.xlsx'.format(7, 7),
                                  './results/combination {}/combination CM R{}.xlsx'.format(8, 8),
                                  './results/combination {}/combination CM R{}.xlsx'.format(9, 9),
                                  './results/combination {}/combination CM R{}.xlsx'.format(10, 10),
                                  './results/combination {}/combination CM R{}.xlsx'.format(11, 11),
                                  './results/combination {}/combination CM R{}.xlsx'.format(12, 12),
                                  './results/combination {}/combination CM R{}.xlsx'.format(13, 13)
                                  ],
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
                                  './excel/Data_loader_spline_full_onehot_R{}_cut_CM3.xlsx'.format(13, 13)]
                                 ,
                                 rounds=[1,2,3,4,5,6,'6e',7,8,9,10,11,12,13],
                                 fn=6, numel=3
                                 )

'''
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
                                  './results/combination {}/models'.format(11)]
'''

#test(9)
test(8)
#test(6, number=1)
#test(6, number=2)
#test(6, number=3)
#test(6, number=4)
#test(6, number=5)
#test(6, number=6)
#test(6, number='6e')
#test(6, number=7)
#test(6, number=8)
#test(6, number=9)
#test(6, number=10)
#test(6, number=11)
#test(6, number=12)
#test(6, number=13)
'''
test(6, number=2)
test(6, number=3)
test(6, number=4)
test(6, number=5)
test(6, number=6)
test(6, number=7)
test(6, number=8)
test(6, number='6e')
model_store = load_model_ensemble('./save/models')
fl = load_data_to_fl('./excel/Data_loader_poly4_caa_090219.xlsx')

prediction = model_store[1].predict(fl.features_c_norm[0,:].reshape((1,-1)))
prediction = [x.item() for x in prediction]
print(prediction)
'''

'''
class CrossStitchLayer(Layer):
    def __init__(self, **kwargs):
        super(CrossStitchLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape should be a list, since cross stitch must take in inputs from all the individual tasks.
        self._input_count = len(input_shape)
        w = np.identity(self._input_count) * 0.9
        inverse_diag_mask = np.invert(np.identity(self._input_count, dtype=np.bool))
        off_value = 0.1 / (self._input_count - 1)
        w[inverse_diag_mask] = off_value
        self._w = K.variable(np.array(w))
        self.trainable_weights.append(self._w)

        super(CrossStitchLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        x = K.stack(x, axis=1)
        y1 = K.dot(self._w, x)
        y = K.permute_dimensions(y1, pattern=(1, 0, 2))
        results = []
        for idx in range(self._input_count):
            results.append(y[:, idx, :])
        return results

    def compute_output_shape(self, input_shape):
        return input_shape

inp = [Input(shape=(5,)), Input(shape=(5,))]
out = CrossStitchLayer()(inp)
model = Model(input=inp, output=out)
model.summary()

a = [np.array([5,1,5,1,5]).reshape((1,-1)), np.array([1,2,3,4,5]).reshape((1,-1))]


output = model.predict(a)
print(output)
'''
