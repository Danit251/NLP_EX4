import pickle
import os
from collections import defaultdict
from itertools import product
from typing import Tuple, List, Dict
from common import PERSON, ORG, TEXT, RELATION
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from data_processor import ProcessAnnotatedData
from relation_vectorizer import RelationsVectorizer

from itertools import chain
from operator import methodcaller

class MlPipe:
    np.random.seed(42)
    models = {'xgboost': XGBClassifier,
              'logreg': LogisticRegression,
              'sgd': SGDClassifier,
              'rf': RandomForestClassifier,
              'svc': LinearSVC}

    def __init__(self, model, **kwargs):
        if model not in self.models:
            raise ValueError(f"Not supported model type: {model} choose from: xgboost, logreg, sgd, rf")
        if 'max_iter' in kwargs:
            self.model = self.models[model](max_iter=kwargs['max_iter'])
        elif 'n_estimators' in kwargs:
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


class RuleBasedpipe:

    def filter_data_set(self, data):
        self._noun_chunk_rule(data)
        return data

    def pred(self, data):
        predictions_noun_chunks = self._noun_chunk_rule(data)
        predictions_bla_bla = self._apposition_rule(data)
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
        for sent in data.i2sentence.values():
            if len(sent.entities[PERSON]) == 0 or len(sent.entities[ORG]) == 0:
                continue
            matched = []
            for chunk in (list(sent.analyzed.noun_chunks)):
                for (per, org) in product(sent.entities[PERSON], sent.entities[ORG]):
                    # str(chunk)
                    if org['text'] in chunk.text and \
                       per['text'] in chunk.text and \
                       per['text'] not in org['text'] and \
                       org['text'] not in per['text']:
                        matched.extend([("PER", per['text']), ("ORG", org['text'])])
                        pred_gen[sent.idx].append((per['text'], org['text'], sent.text))
            sent.entities = self.remove_matched_entities(sent.entities, matched)
        return pred_gen

    @staticmethod
    def remove_matched_entities(unfiltered_ents, matched):
        pass # TODO take down per and org only if there only 1 from each
        if len(matched) == 0:
            return unfiltered_ents
        matched_per = [ent for (t, ent) in matched if t == PERSON]
        matched_org = [ent for (t, ent) in matched if t == ORG]

        filtered_ents = {}
        filtered_ents[PERSON] = [e for e in unfiltered_ents[PERSON] if e['text'] not in matched_per]
        filtered_ents[ORG] = [e for e in unfiltered_ents[ORG] if e['text'] not in matched_org]
        return filtered_ents

    def _apposition_rule(self, data) -> Dict[str, list]:
        all_preds = defaultdict(list)
        for sent in data.i2sentence.values():
            for (cand_per, cand_org) in product(sent.entities['PER'], sent.entities['ORG']):
                org_head_token = self.get_entity_head(sent.analyzed, cand_org)
                if not org_head_token:
                    print(f"can't find org head: {cand_org}")
                    continue

                if self.is_work_for(org_head_token, cand_per):
                    all_preds[sent.idx].append((cand_per["text"], cand_org["text"], sent.text))
        return all_preds

    @staticmethod
    def is_person_token(token, person):
        if person["span"][1] <= token.i <= person["span"][1]:
            return True
        return False

    @staticmethod
    def get_entity_head(doc, entity):
        for j in range(entity["span"][0], entity["span"][1] + 1):
            if doc[j].head.i > entity["span"][1] or doc[j].head.i < entity["span"][0]:
                return doc[j]
        return None

    def is_work_for(self, org_head_token, person):
        if org_head_token.dep_ == "pobj" and org_head_token.head.dep_ == "prep":
            if self.is_person_token(org_head_token.head.head, person):
                return True
            elif org_head_token.head.head.dep_ == "appos" and self.is_person_token(org_head_token.head.head.head, person):
                return True
        elif org_head_token.dep_ in ["nmod", "compound"] and org_head_token.head.dep_ == "appos" and self.is_person_token(
                org_head_token.head.head, person):
            return True
        return False


class RelationExtractionPipeLine:

    TRAIN_F = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cache', "train_data.pkl")
    TEST_F = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'cache', "test_data.pkl")
    MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'cache', "model.pkl")

    def __init__(self):
        self.rb_model = RuleBasedpipe()

        # self.MODEL_PATH = None
        # if train_path:
        #
        #     train, self.test = self.read_train_data(test_path, train_path, use_cache)
        #     self.train = self.rb_model.filter_data_set(train)

        # self.pred = self.rb_model.pred(self.test)
        # self.relation_vectors = RelationsVectorizer(self.train, self.test)
        # self.ml_model = MlPipe('xgboost', n_estimators=1000)
        # self.run()

    def run_train_pipeline(self, train_path, test_path, use_cache=False):
        train, test = self.read_train_data(test_path, train_path, use_cache)
        rb_train_pred = self.rb_model.pred(train)
        rb_test_pred = self.rb_model.pred(test)
        relation_vectors = RelationsVectorizer(train, test)
        model = self.train_model(relation_vectors)
        ml_train_pred = model.predict(relation_vectors.train_vectors)
        self.write_annotated_file(f"PRED.TRAIN.annotations_{model.model_name}.txt", train.op_relations, ml_train_pred, rb_train_pred)
        ml_test_pred = model.predict(relation_vectors.test_vectors)
        self.write_annotated_file(f"PRED.DEV.annotations_{model.model_name}.txt", test.op_relations, ml_test_pred, rb_test_pred)

    @staticmethod
    def write_annotated_file(f_name, op_relations, predicted_labels, rule_based_relations):
        rel_set = set()
        with open(f_name, "w") as f_res:
            for idx, sent_relations in rule_based_relations.items():
                for (person, org, sentence) in sent_relations:
                    rel_str = "_".join([idx, person, RELATION, org])
                    if rel_str not in rel_set:
                        f_res.write("\t".join([idx, person, RELATION, org, f"( {sentence} )\n"]))
                        rel_set.add(rel_str)

            for i, (idx, person, org, sentence) in enumerate(op_relations):
                rel_str = "_".join([idx, person[TEXT], RELATION, org[TEXT]])
                if predicted_labels[i] == "1" and rel_str not in rel_set:
                    f_res.write("\t".join([idx, person[TEXT], RELATION, org[TEXT], f"( {sentence} )\n"]))
                    rel_set.add(rel_str)

    def train_model(self, relation_vectors):
        ml_model = MlPipe('xgboost', n_estimators=1000)
        # ml_model = MlPipe('svc', max_iter=10000)
        ml_model.train_model(relation_vectors.train_vectors, relation_vectors.train_labels)
        # if write_res:
        #     predicted_labels_test = ml_model.predict(relation_vectors.test_vectors)
        #     predicted_labels_train = ml_model.predict(relation_vectors.train_vectors)
        #     write_results(f"PRED.annotations_{ml_model.model_name}.txt", test.op_relations, predicted_labels)
        self.save_to_pickle(ml_model)
        return ml_model




    def run(self):
        pass
        # self.ml_model.train_model(self.relation_vectors.train_vectors, self.relation_vectors.train_labels)
        # predicted_labels = self.ml_model.predict(self.relation_vectors.test_vectors)
        # write_results(f"PRED.annotations_{self.ml_model.model_name}.txt", self.test.op_relations, predicted_labels)

    def read_train_data(self, test_path, train_path, use_cache) -> Tuple[ProcessAnnotatedData, ProcessAnnotatedData]:
        if use_cache:
            train = self.load_from_pickle(self.TRAIN_F)
            test = self.load_from_pickle(self.TEST_F)
            return train, test
        train = ProcessAnnotatedData(train_path)
        self.save_to_pickle(train, self.TRAIN_F)
        test = ProcessAnnotatedData(test_path)
        self.save_to_pickle(test, self.TEST_F)
        return train, test

    def save_to_pickle(self, data):
        with open(self.MODEL_PATH, 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_pickle(f_name):
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
        return data


