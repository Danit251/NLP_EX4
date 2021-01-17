import pickle
import os
from collections import defaultdict
from itertools import product
from typing import Tuple, List, Dict
from common import  PERSON,ORG
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from data_processor import ProcessAnnotatedData
from common import write_results
from relation_vectorizer import RelationsVectorizer

from itertools import chain
from operator import methodcaller

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
            self.model = self.models[model](n_estimators=kwargs['n_estimators']) #, eval_metric='logloss' need to add it and check if its work
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


    def filter_train(self, data):
        self._noun_chunk_rule(data)


    def pred(self, data):
        predictions_noun_chunks = self._noun_chunk_rule(data)
        predictions_bla_bla = self._bla_bla_rule(data)
        # initialise defaultdict of lists
        all_pred = defaultdict(list)

        # iterate dictionary items
        dict_items = map(methodcaller('items'), (predictions_noun_chunks, predictions_bla_bla))
        for k, v in chain.from_iterable(dict_items):
            all_pred[k].extend(v)
        return all_pred





    def _noun_chunk_rule(self, data: ProcessAnnotatedData) -> Dict[str, list]:
        pred_gen = defaultdict(list)
        data: ProcessAnnotatedData
        for sent in data.i2sentence:
            if len(sent.entitis[PERSON]) == 0 or len(sent.entitis[ORG]) == 0:
                continue
            matched = []
            for chunk in (list(sent.analyzed.noun_chunks)):
                for (per, org) in product(sent.entities[PERSON], sent.entities[ORG]):
                    # str(chunk)
                    if org['text'] in chunk.text and \
                       per['text'] in chunk.text and \
                       per['text'] not in org['text'] and \
                       org['text'] not in per['text']:
                        matched.extend([("PER", per), ("ORG", org)])
                        pred_gen[sent.idx].append((per, org))
            sent.entities = self.remove_matched_entities(sent.entitis, matched)
        return pred_gen


    def remove_matched_entities(self, unfiltered_ents, matched):
        if len(matched) == 0:
            return unfiltered_ents
        matched_per = [ent for (t, ent) in matched if t == PERSON]
        matched_org = [ent for (t, ent) in matched if t == ORG]

        filtered_ents = {}
        filtered_ents[PERSON] = [e for e in unfiltered_ents[PERSON] if e['text'] not in matched_per]
        filtered_ents[ORG] = [e for e in unfiltered_ents[ORG] if e['text'] not in matched_org]
        return filtered_ents

    def _bla_bla_rule(self, data)-> Dict[str, list]:
        return {}


class RePipeLine:

    TRAIN_F = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cache', "train_data.pkl")
    TEST_F = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'cache', "test_data.pkl")

    def __init__(self, train_path, test_path, use_cache=False):
        train, test = self.read_data(test_path, train_path, use_cache)
        self.rb_model = RbPipe()
        self.train = self.rb_model.filter_train(train)
        self.test, self.pred = self.rb_model.pred(self.test)
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
