import numpy as np
import numpy.random as rng
import pandas as pd
from openpyxl import load_workbook
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from keras.utils import to_categorical
import pickle
import os
import pathlib
import warnings
import copy
import xlrd
from .others import print_array_to_excel


def load_data_to_fl(data_loader_excel_file, normalise_labels, norm_mask=None):
    df_features = pd.read_excel(data_loader_excel_file, sheet_name='features')
    df_features_d = pd.read_excel(data_loader_excel_file, sheet_name='features_d')
    df_labels = pd.read_excel(data_loader_excel_file, sheet_name='labels')

    features_c = df_features.values
    features_c_names = df_features.columns.values
    if not df_features_d.empty:
        features_d_names = df_features_d.columns.values.tolist()

        # Lookup for features_d descriptors
        lookup_df_store = []
        features_dc_names = []
        for item in features_d_names:
            try:
                temp_df = pd.read_excel(data_loader_excel_file, sheet_name=item)
                temp_df = temp_df.set_index(item)
                lookup_df_store.append(temp_df)
                features_dc_names.extend(temp_df.columns.values.tolist())
            except xlrd.biffh.XLRDError:
                print('{} features_d descriptors not found in excel. Skipping this feature_d'.format(item))

        # Filling up features descriptors for each discrete feature
        features_dc_store = []
        for idx, rows in df_features_d.iterrows():
            row_list = []
            for header, lookup_df in zip(features_d_names, lookup_df_store):
                row_list.extend(lookup_df.loc[rows[header], :].values.tolist())
            features_dc_store.append(row_list)

        # Combine features_c with features_dc
        features_dc_store = np.array(features_dc_store)
        features_c = np.concatenate((features_c, features_dc_store), axis=1)
        features_c_names = np.concatenate((features_c_names, np.array(features_dc_names)), axis=0)
    else:
        lookup_df_store = None

    labels = df_labels.values
    labels_end = labels[:,0][:,None]  # Make 2D array
    labels = labels[:,2:]
    labels_names = df_labels.columns.values

    fl = Features_labels(features_c, labels_end, labels, features_c_names, labels_names, norm_mask=norm_mask,
                         normalise_labels=normalise_labels,
                         features_d_df=df_features_d, lookup_df=lookup_df_store)

    return fl


class Features_labels:
    def __init__(self, features_c, labels_end, labels, features_c_names=None, labels_names=None, scaler=None,
                 norm_mask=None, normalise_labels=False, labels_scaler=None, labels_end_scaler=None,
                 idx=None, features_d_df=None,
                 lookup_df=None):
        """
        Creates fl class with a lot useful attributes
        :param features_c: Continuous features. Np array, no. of examples x continous features
        :param labels: Labels as np array, no. of examples x dim
        :param scaler: Scaler to transform features c. If given, use given MinMax scaler from sklearn,
        else create scaler based on given features c.
        """

        self.features_c_names = features_c_names

        if isinstance(features_d_df, pd.DataFrame):
            if not features_d_df.empty:
                self.features_d_df = features_d_df
                self.lookup_df_store = lookup_df
                self.features_d_names = features_d_df.columns.values.tolist()
                self.features_d_space = []
                for lookup_df in self.lookup_df_store:
                    self.features_d_space.append(lookup_df.index.values.tolist())

        # Setting up features
        self.count = features_c.shape[0]
        if isinstance(idx, np.ndarray):
            self.idx = idx
        else:
            self.idx = np.arange(self.count)
        self.features_c = np.copy(features_c)
        self.features_c_dim = features_c.shape[1]

        if norm_mask:
            self.norm_mask = norm_mask
            mask = np.array([1] * self.features_c_dim, dtype=np.bool)
            mask[norm_mask] = 0
            if features_c[:, mask].shape[1] == 0:
                self.scaler = 0
                self.features_c_norm = features_c
            else:
                if scaler is None:
                    # If scaler is None, means normalize the data with all input data
                    self.scaler = MinMaxScaler()
                    self.scaler.fit(features_c[:, mask])  # Setting up scaler
                else:
                    # If scaler is given, means normalize the data with the given scaler
                    self.scaler = scaler
                features_c_norm = self.scaler.transform(features_c[:, mask])  # Normalizing features_c
                self.features_c_norm = features_c
                self.features_c_norm[:, mask] = features_c_norm
        else:
            # Normalizing continuous features
            self.norm_mask = None
            if scaler is None:
                # If scaler is None, means normalize the data with all input data
                self.scaler = MinMaxScaler()
                self.scaler.fit(features_c)  # Setting up scaler
            else:
                # If scaler is given, means normalize the data with the given scaler
                self.scaler = scaler
            self.features_c_norm = self.scaler.transform(features_c)  # Normalizing features_c

        # Setting up labels
        self.labels_end = labels_end
        self.labels = labels
        if len(labels.shape) == 2:
            self.labels_dim = labels.shape[1]
        else:
            self.labels_dim = 1

        # Label name is size 2 larger than self.labels as it includes the col name for end point and the first pt.
        self.labels_names = labels_names

        # Normalizing labels for the 19 labels going into the neural network
        if normalise_labels:
            self.normalise_labels = normalise_labels
            if labels_scaler is None:
                self.labels_scaler = MinMaxScaler(feature_range=(0, 1))
                self.labels_scaler.fit(labels)
                self.labels_end_scaler = MinMaxScaler(feature_range=(0, 1))
                self.labels_end_scaler.fit(labels_end)
            else:
                self.labels_scaler = labels_scaler
                self.labels_end_scaler = labels_end_scaler
            self.labels_norm = self.labels_scaler.transform(labels)
            self.labels_end_norm = self.labels_end_scaler.transform(labels_end)
        else:
            self.normalise_labels = False
            self.labels_scaler = None
            self.labels_norm = None
            self.labels_end_scaler = None
            self.labels_end_norm = None

    def apply_scaling(self, features_c):
        if features_c.ndim == 1:
            features_c = features_c.reshape((1, -1))
        if self.norm_mask:
            norm_mask = self.norm_mask
            mask = np.array([1] * self.features_c_dim, dtype=np.bool)
            mask[norm_mask] = 0
            features_c_norm = np.copy(features_c)
            features_c_norm[:, mask] = self.scaler.transform(features_c[:, mask])
        else:
            features_c_norm = self.scaler.transform(features_c)
        return features_c_norm

    def generate_random_examples(self, numel):
        gen_features_c_norm_a = rng.random_sample((numel, self.features_c_dim))
        gen_features_c_a = self.scaler.inverse_transform(gen_features_c_norm_a)

        # Creating dic for SNN prediction
        gen_dic = {}
        gen_dic = dict(
            zip(('gen_features_c_a', 'gen_features_c_norm_a'), (gen_features_c_a, gen_features_c_norm_a)))
        return gen_dic

    def create_kf(self, k_folds, shuffle=True):
        '''
        Almost the same as skf except can work for regression labels and folds are not stratified.
        Create list of tuples containing (fl_train,fl_val) fl objects for k fold cross validation
        :param k_folds: Number of folds
        :return: List of tuples
        '''
        fl_store = []
        # Instantiate the cross validator
        skf = KFold(n_splits=k_folds, shuffle=shuffle)
        # Loop through the indices the split() method returns
        for _, (train_indices, val_indices) in enumerate(skf.split(self.features_c, self.labels)):
            # Generate batches from indices
            xval_idx = self.idx[val_indices]
            xtrain, xval = self.features_c[train_indices], self.features_c[val_indices]
            ytrain, yval = self.labels[train_indices], self.labels[val_indices]
            yendtrain, yendval = self.labels_end[train_indices], self.labels_end[val_indices]
            fl_store.append(
                (Features_labels(xtrain, yendtrain, ytrain, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler, labels_end_scaler=self.labels_end_scaler,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names),
                 Features_labels(xval, yendval, yval, idx=xval_idx, scaler=self.scaler, normalise_labels=self.normalise_labels,
                                 labels_scaler=self.labels_scaler, labels_end_scaler=self.labels_end_scaler,
                                 norm_mask=self.norm_mask, features_c_names=self.features_c_names))
            )
        return fl_store

    def create_loocv(self):
        '''
        Create list of tuples containing (fl_train,fl_val) fl objects for leave one out cross validation
        :return: List of tuples
        '''
        fl_store = []
        # Instantiate the cross validator
        loocv = LeaveOneOut()
        # Loop through the indices the split() method returns
        for _, (train_indices, val_indices) in enumerate(loocv.split(self.features_c, self.labels)):
            # Generate batches from indices
            xval_idx = self.idx[val_indices]
            xtrain, xval = self.features_c[train_indices], self.features_c[val_indices]
            ytrain, yval = self.labels[train_indices], self.labels[val_indices]
            fl_store.append(
                (Features_labels(xtrain, ytrain, scaler=self.scaler, labels_scaler=self.labels_scaler,
                                 norm_mask=self.norm_mask),
                 Features_labels(xval, yval, idx=xval_idx, scaler=self.scaler, labels_scaler=self.labels_scaler,
                                 norm_mask=self.norm_mask))
            )
        return fl_store

    def write_data_to_excel(self, loader_excel_file='./excel/data_loader.xlsx'):
        # Excel writing part
        wb = load_workbook(loader_excel_file)
        # Setting up subset features and labels sheet
        sheet_name_store = ['temp_features_c', 'temp_labels']
        ss_store = [self.features_c, self.labels]
        axis_store = [2, 0]  # Because feature_c is 2D while labels is col vector, so order of axis is 2,0
        for cnt, sheet_name in enumerate(sheet_name_store):
            if sheet_name in wb.sheetnames:
                # If temp sheet exists, remove it to create a new one. Else skip this.
                idx = wb.sheetnames.index(sheet_name)  # index of temp sheet
                wb.remove(wb.worksheets[idx])  # remove temp
                wb.create_sheet(sheet_name, idx)  # create an empty sheet using old index
            else:
                wb.create_sheet(sheet_name)  # Create the new sheet
            # Print array to the correct worksheet
            print_array_to_excel(ss_store[cnt], (2, 1), wb[sheet_name_store[cnt]], axis=axis_store[cnt])
        wb.save(loader_excel_file)
        wb.close()


class Features_labels_grid:
    def __init__(self, features, labels, idx=None):
        """
        Creates fl class with a lot useful attributes for grid data classification
        :param features:
        :param labels: Labels as np array, no. of examples x dim
        :param scaler: Scaler to transform features c. If given, use given MinMax scaler from sklearn,
        else create scaler based on given features c.
        """
        # Setting up features
        self.count = features.shape[0]
        self.features = np.copy(features)
        self.features_dim = features.shape[1]

        # idx is for when doing k-fold cross validation, to keep track of which examples are in the val. set
        if isinstance(idx, np.ndarray):
            self.idx = idx
        else:
            self.idx = np.arange(self.count)

        # Setting up labels
        self.labels = np.array(labels)
        self.labels_dim = 1

    def create_kf(self, k_folds, shuffle=True):
        '''
        Almost the same as skf except can work for regression labels and folds are not stratified.
        Create list of tuples containing (fl_train,fl_val) fl objects for k fold cross validation
        :param k_folds: Number of folds
        :return: List of tuples
        '''
        fl_store = []
        # Instantiate the cross validator
        skf = KFold(n_splits=k_folds, shuffle=shuffle)
        # Loop through the indices the split() method returns
        for _, (train_indices, val_indices) in enumerate(skf.split(self.features, self.labels)):
            # Generate batches from indices
            xval_idx = self.idx[val_indices]
            xtrain, xval = self.features[train_indices], self.features[val_indices]
            ytrain, yval = self.labels[train_indices], self.labels[val_indices]
            fl_store.append(
                (Features_labels_grid(xtrain, ytrain),
                 Features_labels_grid(xval, yval, idx=xval_idx)
                 )
            )
        return fl_store