from collections import defaultdict
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, matthews_corrcoef
from tqdm import tqdm
import numpy as np
import pickle
import en_core_web_md
import sys
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
import json
# load spacy
# nlp = en_core_web_md.load()

# load the NER tagger
# tagger = SequenceTagger.load('ner')

RELATION = "Work_For"
PERSON = "PER"
ORG = "ORG"
TEXT = "text"
TYPE = "type"
OFFSETS = "offsets"

TRAIN_F = "train_data.pkl"
TEST_F = "test_data.pkl"
# MODEL_F = f"model.pkl"
LOAD_FROM_PICKLE = True


# class Classifier:
#     def __init__(self, vectorizer):
#         self.model = self.train(vectorizer.train_labels, vectorizer.train_vectors)
#         self.pred_labels = self.predict(vectorizer.test_vectors)


class RelationsVectorizer:

    def __init__(self, train_data, test_data):
        self.dv = DictVectorizer()

        self.train_features = self.get_features(train_data.i2sentence, train_data.relations)
        self.stat = self.create_features_stat()
        self.norm_features(self.train_features)
        self.norm_stat = self.create_features_stat()
        self.test_features = self.get_features(test_data.i2sentence, test_data.relations)

        self.train_vectors = self.dv.fit_transform(self.train_features)
        self.train_labels = np.array(["1"]*len(train_data.pos_relations) + ["0"]*len(train_data.neg_relations))

        self.test_vectors = self.dv.transform(self.test_features)
        self.test_labels = np.array(["1"]*len(test_data.pos_relations) + ["0"]*len(test_data.neg_relations))
        print(self.dv.feature_names_)

    def norm_features(self, features):
        for f_dict in features:
            for f_name, f_val in f_dict.items():
                f_stat = self.stat[f_name][f_val]
                # if f_name in ["dist_sent", "dist_tree"]:
                #     f_val = int(f_val)
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

    def f_pre_word(self, features, person, org, sentence):
        person_start_index = person[OFFSETS][0]
        features["pre_person"] = sentence.analyzed[person_start_index - 1].text

        org_start_index = org[OFFSETS][0]
        features["pre_org"] = sentence.analyzed[org_start_index - 1].text

    def f_pre_pos(self, features, person, org, sentence):
        person_start_index = person[OFFSETS][0]
        features["pre_person_pos"] = sentence.analyzed[person_start_index - 1].pos_

        org_start_index = org[OFFSETS][0]
        features["pre_org_pos"] = sentence.analyzed[org_start_index - 1].pos_
        return features

    def f_cur_word(self, features, person, org, sentence):
        person_start_index = person[OFFSETS][0]
        features["cur_person"] = sentence.analyzed[person_start_index].text

        org_start_index = org[OFFSETS][0]
        features["cur_org"] = sentence.analyzed[org_start_index].text

    def f_cur_pos(self, features, person, org, sentence):
        person_start_index = person[OFFSETS][0]
        features["cur_person_pos"] = sentence.analyzed[person_start_index].pos_

        org_start_index = org[OFFSETS][0]
        features["cur_org_pos"] = sentence.analyzed[org_start_index].pos_
        return features

    def f_after_word(self, features, person, org, sentence):
        person_start_index = person[OFFSETS][0]
        features["after_person"] = sentence.analyzed[person_start_index + 1].text

        org_start_index = org[OFFSETS][0]
        features["after_org"] = sentence.analyzed[org_start_index + 1].text

    def f_after_pos(self, features, person, org, sentence):
        person_start_index = person[OFFSETS][0]
        features["after_person_pos"] = sentence.analyzed[person_start_index + 1].pos_

        org_start_index = org[OFFSETS][0]
        features["after_org_pos"] = sentence.analyzed[org_start_index + 1].pos_

    def f_distance_in_sentence(self, features, person, org, sentence):
        person_start_index = person[OFFSETS][0]
        org_start_index = org[OFFSETS][0]
        features["dist_sent"] = abs(person_start_index - org_start_index)

    def f_distance_in_tree(self, features, person, org, sentence):
        person_start_index = person[OFFSETS][0]
        person_end_index = person[OFFSETS][1]
        org_start_index = org[OFFSETS][0]
        cur_head = sentence.analyzed[org_start_index]
        dist = 1
        while cur_head.dep_ != "ROOT" and (person_start_index > cur_head.i or cur_head.i > person_end_index):
            dist += 1
            cur_head = cur_head.head

        if cur_head.dep_ == "ROOT":
            features["dist_tree"] = 0
        else:
            features["dist_tree"] = dist


class ProcessAnnotatedData:
    def __init__(self, path):
        self.i2sentence, self.i2relations = self.process_data(path)
        self.pos_relations, self.neg_relations = self.get_relations()
        self.relations = self.pos_relations + self.neg_relations

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
                for person, org in sentence.relations:
                    relation = (sentence.idx, person, org, sentence.text)
                    if self.is_relation_pos(gold_rel, person, org, gold_person, gold_org):
                        pos_relations.append(relation)
                    else:
                        neg_relations.append(relation)

        return pos_relations, neg_relations

    def is_relation_pos(self, rel, person, org, arg0, arg1):
        # if rel == RELATION and (person[TEXT] in arg0 or arg0 in person[TEXT]) and\
        #         (org[TEXT] in arg1 or arg1 in org[TEXT]):
        if rel == RELATION and person[TEXT] == arg0 and org[TEXT] == arg1:
            return True
        return False


class RelationSentence:

    def __init__(self, idx, sentence):
        sentence = sentence.replace("-LRB-", "(").replace("-RRB-", ")")
        self.idx = idx
        self.text = sentence
        self.ner = Sentence(sentence)
        self.analyzed = nlp(sentence)
        tagger.predict(self.ner)
        self.entities = [{TEXT: ne.text, TYPE: ne.tag, OFFSETS: (ne.tokens[0].idx - 1, ne.tokens[-1].idx - 1)} for ne in self.ner.get_spans() if ne.tag in [PERSON, ORG]]
        self.relations = self.get_optional_relations()

    def get_optional_relations(self):
        op_relations = []
        persons = [{TEXT: ne[TEXT], OFFSETS: ne[OFFSETS]} for ne in self.entities if ne[TYPE] == PERSON]
        orgs = [{TEXT: ne[TEXT], OFFSETS: ne[OFFSETS]} for ne in self.entities if ne[TYPE] == ORG]

        for person in persons:
            for org in orgs:
                op_relations.append((person, org))

        return op_relations


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
    model = XGBClassifier()
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


model_name = "model_XGB_1000_ww_other"


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
    predicted_labels = predict(model, vectorizer.test_vectors)
    ranked_features = select_features(model, vectorizer.train_vectors, vectorizer.train_labels, vectorizer.dv.feature_names_)
    print(ranked_features)

    report = classification_report(vectorizer.test_labels, predicted_labels)
    print(report)
    mcc = matthews_corrcoef(vectorizer.test_labels, predicted_labels)
    print(f"MCC: {mcc}")
    with open(f"models/report_{model_name}", "w") as f:
        f.write(f"model name: {model_name}\n")
        f.write(report)
        f.write("\n~~~~~~~~~~\n")
        f.write(f"MCC: {mcc}")
        f.write("\n~~~~~~~~~~\n")
        f.write(f"Rank: {ranked_features}")


if __name__ == '__main__':
    main()
