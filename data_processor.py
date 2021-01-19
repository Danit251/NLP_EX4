from itertools import product

import en_core_web_md
from collections import defaultdict
from tqdm import tqdm

from ent_extractor import EntitiesExtraction
from common import RELATION, TEXT, PERSON, ORG


class RelationSentence:
    def __init__(self, idx, text, analyzed, entities):
        self.idx = idx
        self.text = text
        self.analyzed = analyzed
        self.entities = entities

class RelationSentenceBuilder:
    def __init__(self):
        self.nlp = en_core_web_md.load()
        self.ent_extractor = EntitiesExtraction()

    def build_relation_sent(self, idx, sent):
        text = self.clean_sent(sent)
        analyzed = self.nlp(text)
        entities = self.ent_extractor.extract(analyzed)
        return RelationSentence(idx, text, analyzed, entities)

    @staticmethod
    def clean_sent(sent):
        return sent.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LCB-", "").strip("()\n ")

class ProcessCorpusData:
    pass


class ProcessAnnotatedData:
    def __init__(self, path):
        self.relation_builder = RelationSentenceBuilder()
        self.i2sentence, self.i2relations = self.process_data(path)

    def process_data(self, path):
        i2relations = defaultdict(list)
        with open(path) as f:
            lines = f.readlines()
            sentences = {}
            for line in tqdm(lines):
                idx, arg0, relation, arg1, sentence = line.split("\t")
                if idx not in i2relations:
                    sentences[idx] = self.relation_builder.build_relation_sent(idx, sentence)
                i2relations[idx].append((arg0, relation, arg1))

        return sentences, i2relations

    def get_relations(self):
        relations = []
        labels = []
        for sentence in self.i2sentence.values():
            for gold_person, gold_rel, gold_org in self.i2relations[sentence.idx]:
                for person, org in list(product(sentence.entities[PERSON], sentence.entities[ORG])):
                    relations.append((sentence.idx, person, org, sentence.text))
                    if self.is_relation_pos(gold_rel, person, org, gold_person, gold_org):
                        labels.append('1')
                    else:
                        labels.append('0')
        return relations, labels

    def is_relation_pos(self, rel, person, org, arg0, arg1):
        if rel == RELATION and self.is_the_same(person[TEXT], arg0) and self.is_the_same(org[TEXT], arg1):
            return True
        return False

    @staticmethod
    def is_the_same(s1, s2):
        return s1 == s2 or s1 in s2 or s2 in s1
