from collections import defaultdict
from abc import ABC, abstractmethod
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import en_core_web_md
import sys

nlp = en_core_web_md.load()
RELATION = "Work_For"
PERSON = "PERSON"
ORG = "ORG"
TEXT = "text"
TYPE = "type"
OFFSETS = "offsets"


class RelationsVectorizer:

    def __init__(self, train_data, test_data):
        self.dv = DictVectorizer()

        self.train_features = self.get_features(train_data.i2sentence, train_data.relations)
        self.test_features = self.get_features(test_data.i2sentence, test_data.relations)

        self.train_vectors = self.dv.fit_transform(self.train_features)
        self.test_vectors = self.dv.transform(self.test_features)

    def get_features(self, i2sentences, relations):
        features = []
        for relation in relations:
            feature = self.get_relation_features(relation, i2sentences[relation[0]])
            features.append(feature)
        return features

    def get_relation_features(self, relation, sentence):
        features = {}
        for f_feature in [self.f_pre_word, self.f_pre_pos, self.f_after_word, self.f_after_pos]:
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


class ProcessAnnotatedData:
    def __init__(self, path):
        self.i2sentence, self.i2relations = self.process_data(path)
        self.pos_relations, self.neg_relations = self.get_relations()
        self.relations = self.pos_relations + self.neg_relations
        self.labels = np.array(["1"]*len(self.pos_relations) + ["0"]*len(self.neg_relations))

    def process_data(self, path):
        i2relations = defaultdict(list)
        with open(path) as f:
            lines = f.readlines()
            sentences = {}
            for line in lines:
                idx, arg0, relation, arg1, sentence = line.split("\t")
                if idx not in i2relations:
                    sentences[idx] = Sentence(idx, sentence)
                i2relations[idx].append((arg0, relation, arg1))
        return sentences, i2relations

    def get_relations(self):
        pos_relations = []
        neg_relations = []
        for sentence in self.i2sentence.values():
            for arg0, rel, arg1 in self.i2relations[sentence.idx]:
                for person, org in sentence.relations:
                    relation = (sentence.idx, person, org, sentence.text)
                    if self.is_relation_pos(rel, person, org, arg0, arg1):
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


class Sentence:

    def __init__(self, idx, sentence):
        self.idx = idx
        self.text = sentence
        self.analyzed = nlp(sentence)
        self.entities = [{TEXT: ne.text, TYPE: ne.root.ent_type_, OFFSETS: (ne.start, ne.end)} for ne in self.analyzed.ents if ne.root.ent_type_ in [PERSON, ORG]]
        # self.is_candidate_live_in = self.is_candidate_livein()
        # self.is_candidate_work_for = self.is_candidate_work_for()
        self.relations = self.get_optional_relations()

    # def is_candidate_work_for(self):
    #     entities_type = [entity[self.TYPE] for entity in self.entities]
    #     return bool(len(set(entities_type)) == 2)

    def get_optional_relations(self):
        op_relations = []
        persons = [{TEXT: ne[TEXT], OFFSETS: ne[OFFSETS]} for ne in self.entities if ne[TYPE] == PERSON]
        orgs = [{TEXT: ne[TEXT], OFFSETS: ne[OFFSETS]} for ne in self.entities if ne[TYPE] == ORG]

        for person in persons:
            for org in orgs:
                op_relations.append((person, org))

        return op_relations


def main():
    # text = "Israel television rejected a skit by comedian Tuvia Tzafir that attacked public apathy by depicting an Israeli family watching TV while a fire raged outside ."
    # Sentence(text)
    train = ProcessAnnotatedData(sys.argv[1])
    test = ProcessAnnotatedData(sys.argv[2])
    vectorizer = RelationsVectorizer(train, test)


if __name__ == '__main__':
    main()
