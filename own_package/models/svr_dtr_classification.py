from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, mean_squared_error
import pickle, time, gc
import numpy as np
import pandas as pd
from pandas import Series
import openpyxl
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import pycm


class Predict_SVC_DTC:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        # Make sure features are normalized if the model uses features_c_norm and vice versa
        return self.model.predict(features)


class DTCmodel:
    def __init__(self,max_depth=8, num_est=300):
        self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),n_estimators=num_est)

    def train_model(self, fl):
        training_features = fl.features_c_norm
        training_labels = fl.labels_classification
        self.model.fit(training_features, training_labels)
        return self.model

    def eval(self, eval_fl):
        features = eval_fl.features_c_norm
        y_pred = self.model.predict(features)
        return y_pred
