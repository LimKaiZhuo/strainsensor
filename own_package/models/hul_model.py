from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, merge, Input
from keras.engine.topology import Layer
from keras.initializers import Constant
from keras import regularizers
from keras import backend as K
import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, f1_score, matthews_corrcoef, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pprint
import time
from typing import List, Dict
# Own Scripts
from own_package.models.models import cross_stitch, hps


# Custom loss layer
class HULMultiLossLayer(Layer):
    """
    Adapted from: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
    Based on: https://arxiv.org/pdf/1705.07115.pdf
    """

    def __init__(self, nb_outputs=2, init_std=None, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        if isinstance(init_std, list) and np.array(init_std).ndim == 1:
            # Check it is list then check if it also is a list and not nested list
            self.init_std = init_std
        elif isinstance(init_std, np.ndarray) and init_std.ndim == 1:
            # Check if it is a ndarray, then if it is, check if it is 1 dimension.
            self.init_std = init_std.tolist()
        elif not init_std:
            self.init_std = None
        else:
            raise TypeError('init_std must be either list or ndarray of ndim = 1 or None')
        super(HULMultiLossLayer, self).__init__(**kwargs)

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
        super(HULMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs, **kwargs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


class HULMTmodel:
    def __init__(self, fl, mode, hparams, labels_norm=True):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim = [1 for _ in range(fl.labels_dim)]  # Assuming that each task has only 1 dimensional output
        self.hparams = hparams
        self.labels_norm = labels_norm
        features_in = Input(shape=(self.features_dim,), name='main_features_c_input')

        # Selection of model structure
        if mode == 'hps':
            hps_model = hps(self.features_dim, self.labels_dim, self.hparams)
            x = hps_model(features_in)
        elif mode == 'cs':
            cs_model = cross_stitch(self.features_dim, self.labels_dim, self.hparams)
            x = cs_model(features_in)

        # Prediction model to be used for eval and prediction
        self.prediction_model = Model(inputs=features_in, outputs=x, name='Prediction_Model_' + mode)

        # Trainable model to be used for .fit()
        features_in_train = Input(shape=(self.features_dim,), name='train_features_c_input')
        y_predict = self.prediction_model(features_in_train)
        y_true = [Input(shape=(1,)) for _ in range(fl.labels_dim)]
        if self.labels_norm:
            std = fl.labels_norm_std
        else:
            std = fl.labels_std
        out = HULMultiLossLayer(nb_outputs=fl.labels_dim, init_std=std)(y_true + y_predict)
        self.model = Model(inputs=[features_in_train] + y_true, outputs=out, name='Trainable_Model_' + mode)
        self.model.compile(optimizer=hparams['optimizer'], loss=None)

    def train_model(self, fl,
                    save_name='mt.h5', save_dir='./save/models/',
                    save_mode=False, plot_name=None):
        # Training model
        training_features = fl.features_c_norm
        if self.labels_norm:
            training_labels = fl.labels_norm.T.tolist()
        else:
            training_labels = fl.labels.T.tolist()
        training_labels = [np.array(x).reshape(-1, 1) for x in training_labels]
        history = self.model.fit(x=[training_features] + training_labels, y=None,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Debugging check to see features and prediction
        # pprint.pprint(training_features)
        # pprint.pprint(self.prediction_model.predict(training_features))
        # pprint.pprint(training_labels)
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_name:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('HUL loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
        return self.model

    def eval(self, eval_fl):
        features = eval_fl.features_c_norm
        if self.labels_norm:
            labels = eval_fl.labels_norm.tolist()
            labels_actual = eval_fl.labels.tolist()
            predictions = self.prediction_model.predict(features)
            predictions = [prediction.T for prediction in predictions]
            predictions = np.vstack(predictions).T
            predictions = predictions.tolist()
            predictions_actual = eval_fl.labels_scaler.inverse_transform(predictions)
            # Calculating metrics
            mse = mean_squared_error(labels_actual, predictions_actual)
            mse_norm = mean_squared_error(labels, predictions)
        else:
            labels = eval_fl.labels.tolist()
            predictions = self.prediction_model.predict(features)
            predictions = [prediction.T for prediction in predictions]
            predictions = np.vstack(predictions).T
            predictions_actual = predictions.tolist()
            mse = mean_squared_error(labels, predictions_actual)
            mse_norm = mse
        return predictions_actual, mse, mse_norm
