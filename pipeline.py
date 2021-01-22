import pickle
import os
from collections import defaultdict
from itertools import product
from typing import Tuple, Dict

from sklearn.feature_extraction import DictVectorizer

from common import PERSON, ORG, TEXT, RELATION, Relation, is_the_same
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from data_processor import ProcessAnnotatedData, ProcessCorpusData
from relation_vectorizer import RelationsVectorizer
from itertools import chain
from operator import methodcaller


class MlPipe:
    np.random.seed(42)
    models = {'xgboost': XGBClassifier, 'logreg': LogisticRegression, 'sgd': SGDClassifier,
              'rf': RandomForestClassifier, 'svc': LinearSVC}

    def __init__(self, model, **kwargs):
        if model not in self.models:
            raise ValueError(f"Not supported model type: {model} choose from: xgboost, logreg, sgd, rf")
        if 'max_iter' in kwargs:
            self.model = self.models[model](max_iter=kwargs['max_iter'])
        elif 'n_estimators' in kwargs:
            self.model = self.models[model](n_estimators=kwargs['n_estimators'],
                                            scale_pos_weight=kwargs['scale_pos_weight'],
                                            min_child_weight=kwargs['min_child_weight'])

        self.model_name = self.produce_model_name(model, kwargs)

    def produce_model_name(self, model, kwargs):
        kwargs_str = model
        for k, v in kwargs.items():
            kwargs_str += f'{k}_{v}'
        return kwargs_str

    def train_model(self, vectors, labels):
        self.model.fit(vectors, labels)

    def predict(self, test_vectors, op_relations):
        pred = self.model.predict(test_vectors)
        res = defaultdict(list)
        for i, (idx, person, org, sentence) in enumerate(op_relations):
            if pred[i] == "1":
                res[idx].append(Relation(person[TEXT], org[TEXT], sentence))
        return res


class RuleBasedpipe:

    def filter_data_set(self, data):
        self._noun_chunk_rule(data)
        self._apposition_rule(data)
        return data

    def pred(self, data):
        predictions_noun_chunks = self._noun_chunk_rule(data)
        predictions_apposition_rule = self._apposition_rule(data)
        # initialise defaultdict of lists
        all_pred = defaultdict(list)

        # iterate dictionary items
        dict_items = map(methodcaller('items'), (predictions_noun_chunks, predictions_apposition_rule))
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
                        pred_gen[sent.idx].append(Relation(per['text'], org['text'], sent.text))
            self.remove_matched_entities(sent.entities, matched)
        return pred_gen

    @staticmethod
    def remove_matched_entities(unfiltered_ents, matched):
        if len(matched) == 0:
            return unfiltered_ents
        matched_per = [ent for (t, ent) in matched if t == PERSON]
        # We filter only the person and not the org because we assume a person would woek only in one place
        filtered_per = [e for e in unfiltered_ents[PERSON] if e['text'] not in matched_per]
        unfiltered_ents[PERSON] = filtered_per

    def _apposition_rule(self, data) -> Dict[str, list]:
        all_preds = defaultdict(list)
        for sent in data.i2sentence.values():
            matched = []
            for (cand_per, cand_org) in product(sent.entities['PER'], sent.entities['ORG']):
                org_head_token = self.get_entity_head(sent.analyzed, cand_org)
                if not org_head_token:
                    continue
                if self.is_work_for(org_head_token, cand_per):
                    all_preds[sent.idx].append(Relation(cand_per["text"], cand_org["text"], sent.text))
                    matched.extend([("PER", cand_per['text']), ("ORG", cand_org['text'])])
            self.remove_matched_entities(sent.entities, matched)
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
            elif org_head_token.head.head.dep_ == "appos" and self.is_person_token(org_head_token.head.head.head,
                                                                                   person):
                return True
        elif org_head_token.dep_ in {"nmod", "compound"} and \
                org_head_token.head.dep_ == "appos" and \
                self.is_person_token(org_head_token.head.head, person):
            return True
        return False


class RelationExtractionPipeLine:
    TRAIN_F = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cache', "train_data.pkl")
    TEST_F = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cache', "test_data.pkl")
    MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cache', "model.pkl")
    DICT_VECTORIZER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cache', "dv.pkl")
    n_estimators_val = 100
    scale_pos_weight_val = 51
    min_child_weight_val = 2
    def __init__(self):
        self.rb_model = RuleBasedpipe()

    def run_train_pipeline(self, train_path, test_path, use_cache=False, model_grid_search=False, plot_results=False):
        train, test = self.read_train_data(test_path, train_path, use_cache)
        print("Finish Reading data!")
        rb_train_pred = self.rb_model.pred(train)
        rb_test_pred = self.rb_model.pred(test)

        train_op_relations, train_labels = train.get_relations_tag()
        test_op_relations, test_labels = test.get_relations_tag()

        train_vectorized = RelationsVectorizer(train.i2sentence, train_op_relations)
        test_vectorized = RelationsVectorizer(test.i2sentence, test_op_relations, dv=train_vectorized.dv)

        ##########################
        if model_grid_search:
            from eval import main
            for n_estimators_val in range(98, 102):
                for scale_pos_weight_val in range(50, 53):
                    for min_child_weight_val in range(10, 30):
                        min_child_weight_val /= 10
                        xgboost = XGBClassifier(n_estimators=n_estimators_val, scale_pos_weight=scale_pos_weight_val,
                                                min_child_weight=min_child_weight_val)
                        xgboost.fit(train_vectorized.vectors, train_labels)
                        pred = xgboost.predict(test_vectorized.vectors)
                        res = defaultdict(list)
                        for i, (idx, person, org, sentence) in enumerate(test_op_relations):
                            if pred[i] == "1":
                                res[idx].append(Relation(person[TEXT], org[TEXT], sentence))
                        agg = self.aggregate_result(rb_test_pred, res)
                        pred_f_name = f"PRED.TRAIN.annotations_est_{n_estimators_val}_pos_{scale_pos_weight_val}_min_child_weight_{min_child_weight_val}.txt"
                        self.write_annotated_file(pred_f_name, agg)
                        print(
                            f"Reslult for est: {n_estimators_val} pos: {scale_pos_weight_val} child_weight: {min_child_weight_val}")
                        main('data/DEV.annotations.tsv', pred_f_name)
        ###########################


        model = self.train_model(train_vectorized.vectors, train_labels)
        self.save_to_pickle(train_vectorized.dv.vocabulary_, self.DICT_VECTORIZER_PATH)

        ml_train_pred = model.predict(train_vectorized.vectors, train_op_relations)
        train_res = self.aggregate_result(rb_train_pred,
                                          ml_train_pred)


        ml_test_pred = model.predict(test_vectorized.vectors, test_op_relations)
        test_res = self.aggregate_result(rb_test_pred,
                                         ml_test_pred)
        if plot_results:
            self.write_annotated_file(f"PRED.DEV.annotations_{model.model_name}.txt", test_res)
            self.write_annotated_file(f"PRED.TRAIN.annotations_{model.model_name}.txt", train_res)

    def aggregate_result(self, rb_train_pred, ml_train_pred):
        final_res = defaultdict(list)

        for sent_id, rel_list in rb_train_pred.items():
            for rel in rel_list:
                if not self.is_in_final_res(final_res, rel, sent_id):
                    final_res[sent_id].append(rel)

        for sent_id, rel_list in ml_train_pred.items():
            for rel in rel_list:
                if not self.is_in_final_res(final_res, rel, sent_id):
                    final_res[sent_id].append(rel)
        return final_res

    def train_model(self, vectors, labels):
        ml_model = MlPipe('xgboost',
                          n_estimators=self.n_estimators_val,
                          scale_pos_weight=self.scale_pos_weight_val,
                          min_child_weight=self.min_child_weight_val)
        ml_model.train_model(vectors, labels)
        self.save_to_pickle(ml_model, self.MODEL_PATH)
        return ml_model

    def run(self, test_path, output_path):
        test = ProcessCorpusData(test_path)
        model = self.load_from_pickle(self.MODEL_PATH)
        features_names = self.load_from_pickle(self.DICT_VECTORIZER_PATH)
        dv = DictVectorizer()
        dv.fit([features_names])
        rb_test_pred = self.rb_model.pred(test)
        op_relations = test.get_op_relations()
        test_vectorized = RelationsVectorizer(test.i2sentence, op_relations, dv=dv)
        ml_test_pred = model.predict(test_vectorized.vectors, op_relations)
        test_res = self.aggregate_result(rb_test_pred,
                                         ml_test_pred)
        self.write_annotated_file(output_path, test_res)

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

    @staticmethod
    def save_to_pickle(data, path):
        with open(path, 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_pickle(f_name):
        with open(f_name, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def is_in_final_res(final_res, rel: Relation, sent_id):
        for res_rel in final_res[sent_id]:
            new_person = rel.person
            new_org = rel.org
            if is_the_same(res_rel.person, new_person) and is_the_same(res_rel.org, new_org):
                return True
        return False

    @staticmethod
    def write_annotated_file(f_name, train_res):
        with open(f_name, "w") as f_res:
            for idx, sent_relations in train_res.items():
                for (person, org, sentence) in sent_relations:
                    f_res.write("\t".join([idx, person, RELATION, org, f"( {sentence} )\n"]))
