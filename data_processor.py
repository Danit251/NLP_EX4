from itertools import product

import en_core_web_md
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from ent_extractor import EntitiesExtraction
from common import PERSON, ORG, RELATION, TEXT


class RelationSentence:
    def __init__(self, idx, text, analyzed, entities, op_relation):
        self.idx = idx
        self.text = text
        self.analyzed = analyzed
        self.entities = entities
        self.op_relations = op_relation


class RelationSentenceBuilder:
    def __init__(self):
        self.nlp = en_core_web_md.load()
        self.ent_extractor = EntitiesExtraction()

    def build_relation_sent(self, idx, sent):
        text = self.clean_sent(sent)
        analyzed = self.nlp(text)
        entities = self.ent_extractor.extract(text, analyzed)
        op_relation = list(product(entities[PERSON], entities[ORG]))
        return RelationSentence(idx, text, analyzed, entities, op_relation)

    @staticmethod
    def clean_sent(sent):
        return sent.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LCB-", "").strip("()\n ")


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
