from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K
import numpy as np
from own_package.features_labels_setup import load_data_to_fl
from own_package.models.models import create_hparams


# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, init_std=None, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        self.init_std = init_std
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        if self.init_std:
            self.init_std = [np.log(std) for std in self.init_std]
        else:
            self.init_std = [0 for _ in range(self.nb_outputs)]
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(self.init_std[i]), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

sigma1 = 1e1  # ground truth
sigma2 = 1e-2  # ground truth
def gen_data(N):
    X = np.random.randn(N, Q)
    w1 = 2.
    b1 = 8.

    Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, D1)
    w2 = 0.01
    b2 = 0.03

    Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2


N = 50
nb_epoch = 2000
batch_size = 20
nb_features = 10
Q = 1
D1 = 1  # first output
D2 = 1  # second output

def get_prediction_model():
    inp = Input(shape=(Q,), name='inp')
    x = Dense(nb_features, activation='relu')(inp)
    y1_pred = Dense(10, activation='relu')(x)
    y1_pred = Dense(1, activation='linear')(y1_pred)
    y2_pred = Dense(10, activation='relu')(x)
    y2_pred = Dense(1, activation='linear')(y2_pred)
    return Model(inp, [y1_pred, y2_pred])

def get_trainable_model(prediction_model):
    inp = Input(shape=(Q,), name='inp')
    y1_pred, y2_pred = prediction_model(inp)
    y1_true = Input(shape=(D1,), name='y1_true')
    y2_true = Input(shape=(D2,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2, init_std=None)([y1_true, y2_true, y1_pred, y2_pred])
    return Model([inp, y1_true, y2_true], out)

prediction_model = get_prediction_model()
prediction_model.summary()
trainable_model = get_trainable_model(prediction_model)
trainable_model.compile(optimizer='adam', loss=None)
assert len(trainable_model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
assert len(trainable_model.losses) == 1

hparams = create_hparams(shared_layers=[30], ts_layers=[10,10,10], cs_layers=[10,10], epochs=1000,reg_l1=0.001, reg_l2=0.1,
                         activation='relu',batch_size=100, verbose=0)

fl = load_data_to_fl('./excel/Data_loader_test.xlsx', norm_mask=[0])

X = fl.features_c_norm
Y1 = np.copy(fl.labels[:,0])
Y2 = np.copy(fl.labels[:,1])
trainable_model.summary()
hist = trainable_model.fit([X, Y1, Y2], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)

print([np.exp(K.get_value(log_var[0]))**0.5 for log_var in trainable_model.layers[-1].log_vars])
