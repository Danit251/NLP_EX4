from collections import defaultdict
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import pickle
import en_core_web_md
import sys


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
        self.test_features = self.get_features(test_data.i2sentence, test_data.relations)

        self.train_vectors = self.dv.fit_transform(self.train_features)
        self.train_labels = np.array(["1"]*len(train_data.pos_relations) + ["0"]*len(train_data.neg_relations))

        self.test_vectors = self.dv.transform(self.test_features)
        self.test_labels = np.array(["1"]*len(test_data.pos_relations) + ["0"]*len(test_data.neg_relations))

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
        org_start_index = org[OFFSETS][0]
        cur_head = sentence.analyzed[org_start_index]
        dist = 1
        while cur_head.dep_ != "ROOT" and cur_head.idx != person_start_index:
            dist += 1
            cur_head = cur_head.head

        if cur_head.dep_ == "ROOT":
            features["dist_tree"] = len(sentence.analyzed) + 1
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
    model = SGDClassifier(max_iter=1000)
    model.fit(vectors, labels)
    return model


def predict(model, test_vectors):
    pred_labels = []
    for vec in test_vectors:
        pred_labels.append(model.predict(vec)[0])
    return np.array(pred_labels)

model_name = "model_SGD_1000"
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

    report = classification_report(vectorizer.test_labels, predicted_labels)
    print(report)

    with open(f"models/report_{model_name}", "w") as f:
        f.write(report)


if __name__ == '__main__':
    main()
