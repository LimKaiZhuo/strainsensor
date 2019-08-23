import numpy as np
from own_package.preprocess import read_excel_data, line_optimisation
from sklearn.preprocessing import MinMaxScaler
from own_package.active_learning.acquisition import load_model_ensemble
from own_package.features_labels_setup import load_data_to_fl
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.models import Model
import keras
from own_package.active_learning.acquisition import features_to_features_input

'''
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

x=[1,2,3]
print(str(x))

