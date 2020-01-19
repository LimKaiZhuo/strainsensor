import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, merge, Input, concatenate, Reshape, Permute, LSTM,\
    TimeDistributed, RepeatVector, MaxPooling1D, BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import MeanAbsolutePercentageError
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pprint
import time
from typing import List, Dict



def create_hparams(shared_layers=0, ts_layers=0, cs_layers=0,
                   learning_rate=0.001, optimizer='Adam', epochs=100, batch_size=64,
                   activation='relu',
                   shared = 10, end = 10, pre = 10, filters = 10,
                   reg_l1=0, reg_l2=0,
                   epsilon=1, c=1,
                   max_depth=6, num_est=300,
                   verbose=1):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['shared_layers', 'ts_layers', 'cs_layers',
             'learning_rate', 'optimizer', 'epochs', 'batch_size',
             'activation',
             'shared', 'end', 'pre', 'filters',
             'reg_l1', 'reg_l2',
             'epsilon', 'c',
             'max_depth', 'num_est',
             'verbose']
    values = [shared_layers, ts_layers, cs_layers,
              learning_rate, optimizer, epochs, batch_size,
              activation,
              shared, end, pre, filters,
              reg_l1, reg_l2,
              epsilon, c,
              max_depth, num_est,
              verbose]
    hparams = dict(zip(names, values))
    return hparams


def hps(features_dim: int, labels_dim: List[int], hparams: Dict):
    """
    Neural network base model to be called by class Model to build a full network.
    The base model provides a callable keras model that has input of shape specified by features_dim.
    There are multiple output defined by the list provided by labels_dim.
    :param features_dim: Defines the input shape of the callable keras model
    :param labels_dim: Defines how many output there will be and its respective shapes.
    eg: [1, 1, 1, 2] ==> Multi-task learning with 4 separate outputs, each with shape 1, 1, 1, 2 respectively.
    :param hparams: Dict created by create_hparam function. Specifies the various hyperparameters for the neural network
    :return: Callable multi-task keras neural network


    HPS = Hard Parameter Sharing.
    For reference: https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40

    Dense neural network base, till the splitting point where each separate task has its own task specific
    dense neural network.
    Most basic form of multi-task learning.
    """
    # Hyperparameters
    shared_layers = hparams['shared_layers']
    ts_layers = hparams['ts_layers']  # ts stands for task specific
    assert shared_layers, 'shared_layers is an empty list. Ensure that it is not empty as there must at least be 1 ' \
                          'shared dense layer.'
    assert ts_layers, 'ts_layers is an empty list. Ensure that it is not empty as there must at least be 1 task' \
                      'task specific dense layer'

    # Input
    f_in = Input(shape=(features_dim,), name='HPS_Input')

    # Shared layers
    for idx, nodes in enumerate(shared_layers):
        if idx == 0:
            x = Dense(units=nodes,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='shared_' + str(idx))(f_in)
        elif nodes != 0:
            x = Dense(units=nodes,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='shared_' + str(idx))(x)

    # Task Specific Layers
    multi_task = []
    for idx_task, single_task_label in enumerate(labels_dim):
        # Task Specific layers
        for idx_layer, nodes in enumerate(ts_layers):
            if idx_layer == 0:
                single_task = Dense(units=nodes,
                                    activation=hparams['activation'],
                                    kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                                    name='task_' + str(idx_task) + '_layer_' + str(idx_layer))(x)
            else:
                single_task = Dense(units=nodes,
                                    activation=hparams['activation'],
                                    kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                                    name='task_' + str(idx_task) + '_layer_' + str(idx_layer))(single_task)

        # Final output to the correct label dimension
        single_task = Dense(units=single_task_label,
                            activation='linear',
                            kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                            name='task_' + str(idx_task) + '_output_layer')(single_task)
        multi_task.append(single_task)

    # Create input output model
    model = Model(inputs=f_in, outputs=multi_task, name='HPS')
    return model


def ann(features_dim: int, labels_dim: int, hparams: Dict):
    """
    Neural network base model to be called by class Model to build a full network.
    The base model provides a callable keras model that has input of shape specified by features_dim.
    There are multiple output defined by the list provided by labels_dim.
    :param features_dim: Defines the input shape of the callable keras model
    :param labels_dim: Defines how many output there will be and its respective shapes.
    :param hparams: Dict created by create_hparam function. Specifies the various hyperparameters for the neural network
    :return: Callable Keras neural network

    """
    # Hyperparameters
    shared_layers = hparams['shared_layers']
    assert shared_layers, 'shared_layers is an empty list. Ensure that it is not empty as there must at least be 1 ' \
                          'shared dense layer.'

    # Input
    f_in = Input(shape=(features_dim,), name='ANN_Input')

    # Shared layers
    for idx, nodes in enumerate(shared_layers):
        if idx == 0:
            x = Dense(units=nodes,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='shared_' + str(idx))(f_in)
        elif nodes != 0:
            x = Dense(units=nodes,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='shared_' + str(idx))(x)

    # Final output to the correct label dimension
    single_task = Dense(units=labels_dim,
                        activation='linear',
                        kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                        name='output_layer')(x)

    # Create input output model
    model = Model(inputs=f_in, outputs=single_task)
    return model


class Kmodel:
    def __init__(self, fl, mode, hparams):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim = fl.labels_dim  # Assuming that each task has only 1 dimensional output
        self.hparams = hparams
        self.mode = mode
        self.normalise_labels = fl.normalise_labels
        self.labels_scaler = fl.labels_scaler
        features_in = Input(shape=(self.features_dim,), name='main_features_c_input')

        # Selection of model
        if mode == 'ann':
            model = ann(self.features_dim, self.labels_dim, self.hparams)
            x = model(features_in)
            self.model = Model(inputs=features_in, outputs=x)
        elif mode == 'ann2':
            model_1 = ann(self.features_dim, 50, self.hparams)
            x = model_1(features_in)
            model_end = ann(50, 50, self.hparams)
            end = model_end(x)
            end_node = Dense(units=1,
                        activation='linear',
                        kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                        name='output_layer')(end)

            model_2 = ann(50, self.labels_dim-1, self.hparams)

            x = model_2(x)
            self.model = Model(inputs=features_in, outputs=[end_node, x])
        elif mode=='ann3':
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(0))(features_in)
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(1))(x)
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(2))(x)
            # x = BatchNormalization()(x)
            x = Dense(units=self.labels_dim,
                      activation='linear',
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Final')(x)


            self.model = Model(inputs=features_in, outputs=x)
        elif mode=='conv1':
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='shared' + str(1))(features_in)
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(1))(x)
            #x = BatchNormalization()(x)
            x = Dense(units=19,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_set_19')(x)
            #x = BatchNormalization()(x)

            x = Reshape(target_shape=(19, 1))(x)
            x = Conv1D(filters=hparams['filters'], kernel_size=3, strides=1, padding='same', activation='relu')(x)
            #x = BatchNormalization()(x)
            x = Conv1D(filters=hparams['filters']*2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = Conv1D(filters=hparams['filters']*4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            #x = Permute((2,1))(x)
            #x = GlobalAveragePooling1D()(x)
            x = TimeDistributed(Dense(1, activation='linear'))(x)
            x = Reshape(target_shape=(19,))(x)


            self.model = Model(inputs=features_in, outputs=x)

        elif mode=='conv2':
            x = Dense(units=10,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Shared_e_' + str(1))(features_in)
            x = Dense(units=10,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Shared_e_' + str(2))(x)
            end = Dense(units=10,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Dense_e_' + str(1))(x)
            end = Dense(units=10,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Dense_e_' + str(2))(end)
            end_node = Dense(units=1,
                             activation='linear',
                             kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                             name='output_layer')(end)


            x = Dense(units=80,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(1))(x)
            x = Reshape(target_shape=(80, 1))(x)
            x = Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(x)

            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            #x = Permute((2,1))(x)
            #x = GlobalAveragePooling1D()(x)
            x = TimeDistributed(Dense(1, activation='linear'))(x)
            x = Reshape(target_shape=(20,))(x)

            self.model = Model(inputs=features_in, outputs=[end_node, x])

        elif mode=='lstm':
            x = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Shared_e_' + str(1))(features_in)
            x = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Shared_e_' + str(2))(x)
            end = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Dense_e_' + str(1))(x)
            end = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Dense_e_' + str(2))(end)
            end_node = Dense(units=1,
                             activation='linear',
                             kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                             name='output_layer')(end)

            x = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(1))(x)
            x = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(2))(x)

            x = RepeatVector(n=20)(x)
            x = LSTM(units=30, activation='relu', return_sequences=True)(x)
            x = LSTM(units=30, activation='relu', return_sequences=True)(x)

            x = TimeDistributed(Dense(1))(x)
            x = Reshape(target_shape=(20,))(x)
            '''
            x = Permute((2,1))(x)
            x = GlobalAveragePooling1D()(x)
            '''
            self.model = Model(inputs=features_in, outputs=[end_node, x])

        optimizer = Adam(clipnorm=1)
        def weighted_mse(y_true, y_pred):
            loss_weights = np.sqrt(np.arange(1, 20))
            #loss_weights = np.arange(1, 20)
            return K.mean(K.square(y_pred - y_true)*loss_weights, axis=-1)

        self.model.compile(optimizer=optimizer, loss=MeanAbsolutePercentageError())
        #self.model.summary()

    def train_model(self, fl, i_fl,
                    save_name='mt.h5', save_dir='./save/models/',
                    save_mode=False, plot_name=None):
        # Training model
        training_features = fl.features_c_norm
        val_features = i_fl.features_c_norm
        if self.normalise_labels:
            training_labels = fl.labels_norm
            val_labels = i_fl.labels_norm
        else:
            training_labels = fl.labels
            val_labels = i_fl.labels

        # Plotting
        if plot_name:
            history = self.model.fit(training_features, training_labels,
                                     validation_data=(val_features, val_labels),
                                     epochs=self.hparams['epochs'],
                                     batch_size=self.hparams['batch_size'],
                                     verbose=self.hparams['verbose'])
            # Debugging check to see features and prediction
            # pprint.pprint(training_features)
            # pprint.pprint(self.model.predict(training_features))
            # pprint.pprint(training_labels)

            # summarize history for accuracy
            plt.semilogy(history.history['loss'], label=['train'])
            plt.semilogy(history.history['val_loss'], label=['test'])
            plt.plot([],[],' ',label='Final train: {:.3e}'.format(history.history['loss'][-1]))
            plt.plot([], [], ' ', label='Final val: {:.3e}'.format(history.history['val_loss'][-1]))
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
        else:
            history = self.model.fit(training_features, training_labels,
                                     epochs=self.hparams['epochs'],
                                     batch_size=self.hparams['batch_size'],
                                     verbose=self.hparams['verbose'])

        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)

        return self.model, history

    def eval(self, eval_fl):
        features = eval_fl.features_c_norm
        predictions = self.model.predict(features)
        if self.normalise_labels:
            mse_norm = mean_squared_error(eval_fl.labels_norm, predictions)
            mse = mean_squared_error(eval_fl.labels, self.labels_scaler.inverse_transform(predictions))
        else:
            mse = mean_squared_error(eval_fl.labels, predictions)
            mse_norm = mse
        return predictions, mse, mse_norm




class Pmodel:
    def __init__(self, fl, mode, hparams):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        # self.features_dim = fl.features_c_dim
        # self.labels_dim = fl.labels_dim  # Assuming that each task has only 1 dimensional output
        self.features_dim = fl.features_c_dim + 1  # 1 for the positional argument
        self.labels_dim = 1
        self.numel = fl.labels.shape[1] + 1
        self.hparams = hparams
        self.mode = mode
        self.normalise_labels = fl.normalise_labels
        self.labels_scaler = fl.labels_scaler
        features_in = Input(shape=(self.features_dim,), name='main_features_c_input')

        # Selection of model
        if mode == 'ann':
            model = ann(self.features_dim, self.labels_dim, self.hparams)
            x = model(features_in)
            self.model = Model(inputs=features_in, outputs=x)
        elif mode == 'ann2':
            model_1 = ann(self.features_dim, 50, self.hparams)
            x = model_1(features_in)
            model_end = ann(50, 50, self.hparams)
            end = model_end(x)
            end_node = Dense(units=1,
                        activation='linear',
                        kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                        name='output_layer')(end)

            model_2 = ann(50, self.labels_dim-1, self.hparams)

            x = model_2(x)
            self.model = Model(inputs=features_in, outputs=[end_node, x])
        elif mode=='ann3':
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(0))(features_in)
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(1))(x)
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(2))(x)
            # x = BatchNormalization()(x)
            x = Dense(units=1,
                      activation='linear',
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_set_19')(x)


            self.model = Model(inputs=features_in, outputs=x)
        elif mode=='conv1':
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='shared' + str(1))(features_in)
            x = Dense(units=hparams['pre'],
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(1))(x)
            #x = BatchNormalization()(x)
            x = Dense(units=19,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_set_19')(x)
            #x = BatchNormalization()(x)

            x = Reshape(target_shape=(19, 1))(x)
            x = Conv1D(filters=hparams['filters'], kernel_size=3, strides=1, padding='same', activation='relu')(x)
            #x = BatchNormalization()(x)
            x = Conv1D(filters=hparams['filters']*2, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = Conv1D(filters=hparams['filters']*4, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            #x = Permute((2,1))(x)
            #x = GlobalAveragePooling1D()(x)
            x = TimeDistributed(Dense(1, activation='linear'))(x)
            x = Reshape(target_shape=(19,))(x)


            self.model = Model(inputs=features_in, outputs=x)

        elif mode=='conv2':
            x = Dense(units=10,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Shared_e_' + str(1))(features_in)
            x = Dense(units=10,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Shared_e_' + str(2))(x)
            end = Dense(units=10,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Dense_e_' + str(1))(x)
            end = Dense(units=10,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Dense_e_' + str(2))(end)
            end_node = Dense(units=1,
                             activation='linear',
                             kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                             name='output_layer')(end)


            x = Dense(units=80,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(1))(x)
            x = Reshape(target_shape=(80, 1))(x)
            x = Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(x)

            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            #x = Permute((2,1))(x)
            #x = GlobalAveragePooling1D()(x)
            x = TimeDistributed(Dense(1, activation='linear'))(x)
            x = Reshape(target_shape=(20,))(x)

            self.model = Model(inputs=features_in, outputs=[end_node, x])

        elif mode=='lstm':
            x = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Shared_e_' + str(1))(features_in)
            x = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Shared_e_' + str(2))(x)
            end = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Dense_e_' + str(1))(x)
            end = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Dense_e_' + str(2))(end)
            end_node = Dense(units=1,
                             activation='linear',
                             kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                             name='output_layer')(end)

            x = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(1))(x)
            x = Dense(units=20,
                      activation=hparams['activation'],
                      kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                      name='Pre_' + str(2))(x)

            x = RepeatVector(n=20)(x)
            x = LSTM(units=30, activation='relu', return_sequences=True)(x)
            x = LSTM(units=30, activation='relu', return_sequences=True)(x)

            x = TimeDistributed(Dense(1))(x)
            x = Reshape(target_shape=(20,))(x)
            '''
            x = Permute((2,1))(x)
            x = GlobalAveragePooling1D()(x)
            '''
            self.model = Model(inputs=features_in, outputs=[end_node, x])

        optimizer = Adam(clipnorm=1)

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        #self.model.summary()

    def train_model(self, fl, i_fl,
                    save_name='mt.h5', save_dir='./save/models/',
                    save_mode=False, plot_name=None):
        # Training model
        training_features = fl.features_c_norm
        val_features = i_fl.features_c_norm

        if self.normalise_labels:
            training_labels = fl.labels_norm
            val_labels = i_fl.labels_norm
        else:
            training_labels = fl.labels
            val_labels = i_fl.labels

        p_features = []
        for features in training_features.tolist():
            for idx in list(range(1,self.numel)):
                p_features.append(features+[idx])

        training_features = np.array(p_features)

        training_labels = training_labels.flatten()[:, None]

        # Plotting
        if plot_name:
            p_features = []
            for features in val_features.tolist():
                for idx in list(range(1, self.numel)):
                    p_features.append(features + [idx])

            val_features = np.array(p_features)

            val_labels = val_labels.flatten()[:, None]


            history = self.model.fit(training_features, training_labels,
                                     validation_data=(val_features, val_labels),
                                     epochs=self.hparams['epochs'],
                                     batch_size=self.hparams['batch_size'],
                                     verbose=self.hparams['verbose'])
            # Debugging check to see features and prediction
            # pprint.pprint(training_features)
            # pprint.pprint(self.model.predict(training_features))
            # pprint.pprint(training_labels)

            # summarize history for accuracy
            plt.semilogy(history.history['loss'], label=['train'])
            plt.semilogy(history.history['val_loss'], label=['test'])
            plt.plot([],[],' ',label='Final train: {:.3e}'.format(history.history['loss'][-1]))
            plt.plot([], [], ' ', label='Final val: {:.3e}'.format(history.history['val_loss'][-1]))
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
        else:
            history = self.model.fit(training_features, training_labels,
                                     epochs=self.hparams['epochs'],
                                     batch_size=self.hparams['batch_size'],
                                     verbose=self.hparams['verbose'])

        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)

        return self.model, history

    def eval(self, eval_fl):
        eval_features = eval_fl.features_c_norm

        predictions = []
        for features in eval_features.tolist():
            single_expt = []
            for idx in list(range(1,self.numel)):
                single_expt.append(self.model.predict(np.array(features+[idx])[None,...])[0][0])
            predictions.append(single_expt)

        predictions = np.array(predictions)

        if self.normalise_labels:
            mse_norm = mean_squared_error(eval_fl.labels_norm, predictions)
            mse = mean_squared_error(eval_fl.labels, self.labels_scaler.inverse_transform(predictions))
        else:
            mse = mean_squared_error(eval_fl.labels, predictions)
            mse_norm = mse
        return predictions, mse, mse_norm

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
'''

def cross_stitch(features_dim: int, labels_dim: List[int], hparams: Dict):
    """
    Neural network base model to be called by class Model to build a full network.
    The base model provides a callable keras model that has input of shape specified by features_dim.
    There are multiple output defined by the list provided by labels_dim.
    :param features_dim: Defines the input shape of the callable keras model
    :param labels_dim: Defines how many output there will be and its respective shapes.
    eg: [1, 1, 1, 2] ==> Multi-task learning with 4 separate outputs, each with shape 1, 1, 1, 2 respectively.
    :param hparams: Dict created by create_hparam function. Specifies the various hyperparameters for the neural network
    :return: Callable multi-task keras neural network

    Cross-stitch neural network paper
    https://ieeexplore.ieee.org/document/7780802
    """

    cs_layers = hparams['cs_layers']
    assert cs_layers, 'cs_layers is an empty list. Ensure that it is not empty as there must at least be 1 ' \
                      'cross stitch dense layer.'

    # Input
    f_in = Input(shape=(features_dim,), name='CS_Input')

    # Cross stitch layers
    multi_task = []
    for idx, nodes in enumerate(cs_layers):
        if idx == 0:
            for idx_task, _ in enumerate(labels_dim):
                x = Dense(units=nodes,
                          activation=hparams['activation'],
                          kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                          name='cs_task_' + str(idx_task) + '_layer_' + str(idx))(f_in)
                multi_task.append(x)
            multi_task = CrossStitchLayer(name='cs_unit_' + str(idx))(multi_task)

        elif nodes != 0:
            temp = []
            for idx_task, single_task in enumerate(multi_task):
                single_task = Dense(units=nodes,
                                    activation=hparams['activation'],
                                    kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'],
                                                                          l2=hparams['reg_l2']),
                                    name='cs_task_' + str(idx_task) + '_layer_' +
                                         str(idx))(single_task)
                temp.append(single_task)
            multi_task = CrossStitchLayer(name='cs_unit_' + str(idx))(temp)
    temp = []
    for idx_task, single_task_label in enumerate(labels_dim):
        single_task = multi_task[idx_task]
        single_task = Dense(units=single_task_label,
                            activation='linear',
                            kernel_regularizer=regularizers.l1_l2(l1=hparams['reg_l1'], l2=hparams['reg_l2']),
                            name='task_' + str(idx_task) + '_output_layer')(single_task)
        temp.append(single_task)
    multi_task = temp
    model = Model(inputs=f_in, outputs=multi_task, name='CS')
    return model


class MTmodel:
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

        # Selection of model
        if mode == 'hps':
            hps_model = hps(self.features_dim, self.labels_dim, self.hparams)
            x = hps_model(features_in)
        elif mode == 'cs':
            cs_model = cross_stitch(self.features_dim, self.labels_dim, self.hparams)
            x = cs_model(features_in)

        self.model = Model(inputs=features_in, outputs=x)
        self.model.compile(optimizer=hparams['optimizer'], loss='mean_squared_error')

    def train_model(self, fl, i_fl,
                    save_name='mt.h5', save_dir='./save/models/',
                    save_mode=False, plot_name=None):
        # Training model
        training_features = fl.features_c_norm
        if self.labels_norm:
            training_labels = fl.labels_norm.T.tolist()
        else:
            training_labels = fl.labels.T.tolist()

        if plot_name:
            history = self.model.fit(training_features, training_labels,
                                     epochs=self.hparams['epochs'],
                                     batch_size=self.hparams['batch_size'],
                                     verbose=self.hparams['verbose'])
            # Debugging check to see features and prediction
            # pprint.pprint(training_features)
            # pprint.pprint(self.model.predict(training_features))
            # pprint.pprint(training_labels)
            # Saving Model
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
        else:
            self.model.fit(training_features, training_labels,
                           epochs=self.hparams['epochs'],
                           batch_size=self.hparams['batch_size'],
                           verbose=self.hparams['verbose'])

        if save_mode:
            self.model.save(save_dir + save_name)

        return self.model

    def eval(self, eval_fl):
        features = eval_fl.features_c_norm
        if self.labels_norm:
            labels = eval_fl.labels_norm.tolist()
            labels_actual = eval_fl.labels.tolist()
            predictions = self.model.predict(features)
            predictions = [prediction.T for prediction in predictions]
            predictions = np.vstack(predictions).T
            predictions = predictions.tolist()
            predictions_actual = eval_fl.labels_scaler.inverse_transform(predictions)
            # Calculating metrics
            mse = mean_squared_error(labels_actual, predictions_actual)
            mse_norm = mean_squared_error(labels, predictions)
        else:
            labels = eval_fl.labels.tolist()
            predictions = self.model.predict(features)
            predictions = [prediction.T for prediction in predictions]
            predictions = np.vstack(predictions).T
            predictions_actual = predictions.tolist()
            mse = mean_squared_error(labels, predictions_actual)
            mse_norm = mse
        return predictions_actual, mse, mse_norm
