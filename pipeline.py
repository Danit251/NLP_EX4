import pickle
import os
from typing import Tuple

import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from data_processor import ProcessAnnotatedData
from common import write_results
from relation_vectorizer import RelationsVectorizer


class MlPipe:
    np.random.seed(42)
    models = {'xgboost': XGBClassifier,
              'logreg': LogisticRegression,
              'sgd': SGDClassifier,
              'rf': RandomForestClassifier}

    def __init__(self, model, **kwargs):
        if model not in self.models:
            raise ValueError(f"Not supported model type: {model} choose from: xgboost, logreg, sgd, rf")
        if 'n_estimators' in kwargs:
            self.model = self.models[model](n_estimators=kwargs['n_estimators'])
        self.model_name = self.produce_model_name(model, kwargs)

    def produce_model_name(self, model, kwargs):
        kwargs_str = model
        for k, v in kwargs.items():
            kwargs_str += f'{k}_{v}'
        return kwargs_str




    def train_model(self,  vectors, labels):
        # model = RandomForestClassifier(n_estimators=1000)
        # model = LogisticRegression(max_iter=1000)
        # model = SGDClassifier(max_iter=1000)
        # model = XGBClassifier(n_estimators=1000)
        self.model.fit(vectors, labels)


    def predict(self, test_vectors):
        return self.model.predict(test_vectors)


class RbPipe:
    def __init__(self, test_dat):
        pass


class RePipeLine:

    TRAIN_F = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cache', "train_data.pkl")
    TEST_F = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'cache', "test_data.pkl")

    def __init__(self, train_path, test_path, use_cache=False):
        self.train, self.test = self.read_data(test_path, train_path, use_cache)
        # self.rb_model = RbPipe()
        self.relation_vectors = RelationsVectorizer(self.train, self.test)
        self.ml_model = MlPipe('xgboost', n_estimators=1000)
        self.run()

    def run(self):
        self.ml_model.train_model(self.relation_vectors.train_vectors, self.relation_vectors.train_labels)
        predicted_labels = self.ml_model.predict(self.relation_vectors.test_vectors)
        write_results(f"PRED.annotations_{self.ml_model.model_name}.txt", self.test.op_relations, predicted_labels)

    def read_data(self, test_path, train_path, use_cache) ->  Tuple[ProcessAnnotatedData, ProcessAnnotatedData]:
        if use_cache:
            train = self.load_from_pickle(self.TRAIN_F)
            test = self.load_from_pickle(self.TEST_F)
            return train, test
        train = ProcessAnnotatedData(train_path)
        self.save_to_pickle(train, self.TRAIN_F)
        test = ProcessAnnotatedData(test_path)
        self.save_to_pickle(test, self.TEST_F)
        return train, test

    @staticmethod
    def save_to_pickle(data, f_name):
        with open(f_name, 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_pickle(f_name):
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
        return data
