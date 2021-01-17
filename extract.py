from collections import defaultdict
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import pickle
import en_core_web_md
import sys
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
import json
from extractors import SpacyExtractor, FlairExtractor
from itertools import product

# load spacy
nlp = en_core_web_md.load()

RELATION = "Work_For"
PERSON = "PER"
ORG = "ORG"
TEXT = "text"
TYPE = "type"
SPAN = "span"

TRAIN_F = "train_data_embedding.pkl"
TEST_F = "test_data_embedding.pkl"
LOAD_FROM_PICKLE = True
model_name = "model_XGB_1000_feat_deps"


class WeVectorizer:
    def __init__(self,  train_data, test_data):
        self.vectorizer = en_core_web_md.load()
        self.train_vec = self.vectorizer_data(train_data.op_relations)
        self.train_labels = train_data.labels

        self.test_vec = self.vectorizer_data(test_data.op_relations)
        self.test_labels = test_data.labels

    def vectorizer_data(self, relations):
        vecs = []
        for sent_id, per_cand, org_cand, sent_raw  in relations:
            sent = sent_raw.strip("().\n")
            org = org_cand['text']
            per = per_cand['text']
            sent_clean = sent.replace(org, "").replace(per, "")
            vecs.append(self.vec_sent(sent_clean, per, org))
        vecs = np.array(vecs)
        return vecs

    def vec_sent(self, sent, per_candidate, org_candidate):
            toks = [t for t in self.vectorizer(sent) if not any([t.is_space, t.is_punct, t.is_stop, t.is_currency]) and t.has_vector]
            sent_vecs = np.array([t.vector for t in toks]).mean(axis=0)
            per_vec = self.vectorize_ent(per_candidate)
            org_vec = self.vectorize_ent(org_candidate)
            res = np.concatenate([sent_vecs, per_vec, org_vec])
            return res

    def vectorize_ent(self, org_candidate):
        return np.array([t.vector for t in self.vectorizer(org_candidate)]).mean(axis=0)


class RelationsVectorizer:

    def __init__(self, train_data, test_data):
        # self.embedding_vectorizer = WeVectorizer(train_data, test_data)
        self.dv = DictVectorizer()

        self.train_features = self.get_features(train_data.i2sentence, train_data.op_relations)
        self.stat = self.create_features_stat()
        self.norm_features(self.train_features)
        # self.norm_stat = self.create_features_stat()
        self.test_features = self.get_features(test_data.i2sentence, test_data.op_relations)

        # self.e_train_vectors = self.embedding_vectorizer.train_vec
        # self.f_train_vectors = self.dv.fit_transform(self.train_features).toarray()
        # self.train_vectors = self.merge_vectors(self.f_train_vectors, self.e_train_vectors)
        self.train_vectors = self.dv.fit_transform(self.train_features).toarray()
        self.train_labels = train_data.labels

        # self.e_test_vectors = self.embedding_vectorizer.test_vec
        # self.f_test_vectors = self.dv.transform(self.test_features).toarray()
        # self.test_vectors = self.merge_vectors(self.f_test_vectors, self.e_test_vectors)
        self.test_vectors = self.dv.transform(self.test_features).toarray()
        self.test_labels = test_data.labels
        print(self.dv.feature_names_)

    def merge_vectors(self, feat_vectors, embedding_vectors):
        merged = []
        for f_vec, e_vec in zip(feat_vectors, embedding_vectors):
            merged.append(np.concatenate([f_vec, e_vec]))
        return np.array(merged)

    def norm_features(self, features):
        for f_dict in features:
            for f_name, f_val in f_dict.items():
                f_stat = self.stat[f_name][f_val]
                if (f_name == "dist_sent" and (f_val < 4 or f_val > 13)) or \
                        (f_name == "dist_tree" and (f_val < 3 or f_val > 6)):
                    f_dict[f_name] = 0
                if f_name not in ["dist_sent", "dist_tree"] and f_stat < 20:
                    f_dict[f_name] = "OTHER"

    def create_features_stat(self):
        stat = defaultdict(lambda: defaultdict(int))
        for f in self.train_features:
            for f_name, f_val in f.items():
                stat[f_name][f_val] += 1

        for f_name, f_stat in stat.items():
            stat[f_name] = {k: v for k, v in sorted(f_stat.items(), key=lambda item: item[1], reverse=True)}

        return stat

    def get_features(self, i2sentences, relations):
        features = []
        for relation in relations:
            feature = self.get_relation_features(relation, i2sentences[relation[0]])
            features.append(feature)
        return features

    def get_relation_features(self, relation, sentence):
        features = {}
        for f_feature in [self.f_pre_pos, self.f_after_pos, self.f_distance_in_sentence, self.f_distance_in_tree,
                          self.f_cur_pos]:
            f_feature(features, relation[1], relation[2], sentence)
        return features

    def f_pre_pos(self, features, person, org, sentence):
        person_start_index = person[SPAN][0]
        if person_start_index != 0:
            features["pre_person_pos"] = sentence.analyzed[person_start_index - 1].pos_
        else:
            features["pre_person_pos"] = "START"

        org_start_index = org[SPAN][0]
        if org_start_index != 0:
            features["pre_org_pos"] = sentence.analyzed[org_start_index - 1].pos_
        else:
            features["pre_org_pos"] = "START"

        return features

    def f_cur_pos(self, features, person, org, sentence):
        person_start_index = person[SPAN][0]
        features["cur_person_pos"] = sentence.analyzed[person_start_index].pos_

        org_start_index = org[SPAN][0]
        features["cur_org_pos"] = sentence.analyzed[org_start_index].pos_
        return features

    def f_after_pos(self, features, person, org, sentence):
        person_end_index = person[SPAN][1]
        try:
            if person_end_index != len(sentence.analyzed.doc) - 1:
                features["after_person_pos"] = sentence.analyzed[person_end_index + 1].pos_
            else:
                features["after_person_pos"] = "END"
        except Exception as error:
            pass

        org_end_index = org[SPAN][1]
        if org_end_index != len(sentence.analyzed.doc) - 1:
            features["after_org_pos"] = sentence.analyzed[org_end_index + 1].pos_
        else:
            features["after_org_pos"] = "END"

    def f_distance_in_sentence(self, features, person, org, sentence):
        person_start_index = person[SPAN][0]
        org_start_index = org[SPAN][0]
        features["dist_sent"] = abs(person_start_index - org_start_index)

    def f_distance_in_tree(self, features, person, org, sentence):
        person_start_index = person[SPAN][0]
        person_end_index = person[SPAN][1]
        org_start_index = org[SPAN][0]
        cur_head = sentence.analyzed[org_start_index]
        dist = 1
        # all_deps = defaultdict(int)
        while cur_head.dep_ != "ROOT" and (person_start_index > cur_head.i or cur_head.i > person_end_index):
            dist += 1
            cur_head = cur_head.head
            # all_deps[f"dep_{cur_head.dep_}"] += 1
        #
        # if cur_head.dep_ != "ROOT":
        #     all_deps[f"dep_{cur_head.dep_}"] += 1
        #     features.update(all_deps)

        if cur_head.dep_ == "ROOT":
            features["dist_tree"] = 0
        else:
            features["dist_tree"] = dist


class ProcessAnnotatedData:
    def __init__(self, path):
        self.i2sentence, self.i2relations = self.process_data(path)
        self.pos_relations, self.neg_relations = self.get_relations()
        self.op_relations = self.pos_relations + self.neg_relations
        self.labels = np.array(["1"] * len(self.pos_relations) + ["0"] * len(self.neg_relations))

    def process_data(self, path):
        i2relations = defaultdict(list)
        with open(path) as f:
            lines = f.readlines()
            sentences = {}
            for line in tqdm(lines):
                idx, arg0, relation, arg1, sentence = line.split("\t")
                if idx not in i2relations:
                    sentences[idx] = RelationSentence(idx, sentence)
                i2relations[idx].append((arg0, relation, arg1))

        return sentences, i2relations

    def get_relations(self):
        pos_relations = []
        neg_relations = []
        for sentence in self.i2sentence.values():
            for gold_person, gold_rel, gold_org in self.i2relations[sentence.idx]:
                for person, org in sentence.op_relations:
                    relation = (sentence.idx, person, org, sentence.text)
                    if self.is_relation_pos(gold_rel, person, org, gold_person, gold_org):
                        pos_relations.append(relation)
                    else:
                        neg_relations.append(relation)

        return pos_relations, neg_relations

    def is_relation_pos(self, rel, person, org, arg0, arg1):
        if rel == RELATION and person[TEXT] == arg0 and org[TEXT] == arg1:
            return True
        return False


def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance


@singleton
class EntitiesExtraction:

    def __init__(self):
        self.ext_spacy = SpacyExtractor()
        self.ext_flair = FlairExtractor()

    def extract(self, sentence, analyzed_sent):
        spacy_entities = self.ext_spacy.extract(sentence)
        flair_entities = self.ext_flair.extract(sentence)
        self.update_offsets(flair_entities, analyzed_sent)
        spacy_entities = self.remove_contradict_entities(spacy_entities, flair_entities)

        res = defaultdict(list)
        for ent_type in [PERSON, ORG]:
            s = spacy_entities[ent_type] if ent_type in spacy_entities else []
            f = flair_entities[ent_type] if ent_type in flair_entities else []

            if len(f) >= len(s):
                res[ent_type] = f
            else:
                for ent_s in s:
                    inserted = False

                    # if finds entity in flair insert it
                    for ent_f in f:
                        if (ent_s[SPAN][1] - ent_s[SPAN][0]) == (ent_f[SPAN][1] - ent_f[SPAN][0]):
                            if ent_s[TEXT] == ent_f[TEXT]:
                                res[ent_type].append(ent_f)
                                inserted = True
                                break
                        else:
                            if ent_s[TEXT] in ent_f[TEXT] or ent_f[TEXT] in ent_s[TEXT]:
                                res[ent_type].append(ent_f)
                                inserted = True
                                break

                    # if didn't find entity in flair insert it from spacy
                    if not inserted:
                        res[ent_type].append(ent_s)

        return res

    def remove_contradict_entities(self, spacy_entities, flair_entities):
        spacy_person = spacy_entities[PERSON] if PERSON in spacy_entities else []
        spacy_org = spacy_entities[ORG] if ORG in spacy_entities else []

        if PERSON in flair_entities:
            person_flair = set([ent[TEXT] for ent in flair_entities[PERSON]])
            if ORG in spacy_entities:
                spacy_org = [s for s in spacy_entities[ORG] if s[TEXT] not in person_flair]

        if ORG in flair_entities:
            org_flair = set([ent[TEXT] for ent in flair_entities[ORG]])
            if PERSON in spacy_entities:
                spacy_person = [s for s in spacy_entities[PERSON] if s[TEXT] not in org_flair]

        return {ORG: spacy_org, PERSON: spacy_person}

    def update_offsets(self, entities, analyzed_sent):
        for type_entity in entities:
            for entity in entities[type_entity]:
                for word in analyzed_sent:
                    if entity[SPAN][0] == word.idx:
                        entity[SPAN] = (word.i, word.i + entity[SPAN][1] - 1)


class RelationSentence:

    def __init__(self, idx, sentence):
        sentence = sentence.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LCB-", "").strip("()\n ")
        self.idx = idx
        self.text = sentence
        self.analyzed = nlp(sentence)
        extractor = EntitiesExtraction()
        self.entities = extractor.extract(sentence, self.analyzed)
        self.op_relations = list(product(self.entities[PERSON], self.entities[ORG]))


def save_to_pickle(data, f_name):
    with open(f_name, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)


def load_from_pickle(f_name):
    with open(f_name, 'rb') as f:
        data = pickle.load(f)
    return data


def train_model(labels, vectors):
    # model = RandomForestClassifier(n_estimators=1000)
    # model = LogisticRegression(max_iter=1000)
    # model = SGDClassifier(max_iter=1000)
    model = XGBClassifier(n_estimators=1000)
    model.fit(vectors, labels)
    return model


def predict(model, test_vectors):
    pred_labels = []
    for vec in test_vectors:
        pred_labels.append(model.predict(vec)[0])
    return np.array(pred_labels)


def select_features(model, vectors, labels, features_names):
    rfe = RFE(model, 1)
    fit = rfe.fit(vectors, labels)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % list(fit.ranking_))
    rank = list(fit.ranking_)
    merge = {item: int(rank[i]) for i, item in enumerate(features_names)}
    f_sorted = {k: v for k, v in sorted(merge.items(), key=lambda item: item[1])}
    return json.dumps(f_sorted, ensure_ascii=False, indent=4)


def write_results(f_name, op_relations, predicted_labels):
    rel_set = set()
    with open(f_name, "w") as f_res:
        for i, (idx, person, org, sentence) in enumerate(op_relations):
            rel_str = "_".join([idx, person[TEXT], RELATION, org[TEXT]])
            if predicted_labels[i] == "1" and rel_str not in rel_set:
                f_res.write("\t".join([idx, person[TEXT], RELATION, org[TEXT], f"( {sentence} )\n"]))
                rel_set.add(rel_str)


def main():
    np.random.seed(42)
    if LOAD_FROM_PICKLE:
        train = load_from_pickle(TRAIN_F)
        test = load_from_pickle(TEST_F)
    else:
        train = ProcessAnnotatedData(sys.argv[1])
        save_to_pickle(train, TRAIN_F)
        test = ProcessAnnotatedData(sys.argv[2])
        save_to_pickle(test, TEST_F)

    vectorizer = RelationsVectorizer(train, test)

    model = train_model(vectorizer.train_labels, vectorizer.train_vectors)
    save_to_pickle(model, f"models/{model_name}.pkl")
    predicted_labels = model.predict(vectorizer.test_vectors)
    write_results(f"PRED.annotations_{model_name}.txt", test.op_relations, predicted_labels)


if __name__ == '__main__':
    main()
