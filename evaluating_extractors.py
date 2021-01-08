from collections import defaultdict
from typing import List, Tuple, Dict
from itertools import groupby
from flair.data import Sentence
from flair.models import SequenceTagger
import en_core_web_md
from abc import ABC, abstractmethod
from tqdm import tqdm

class Extractor(ABC):

    @abstractmethod
    def extract(self, sentence: str) -> Dict[str, List[str]]:
        pass


class SpacyExtractor(Extractor):

    def __init__(self):
        self.nlp = en_core_web_md.load()
        self.valid_entity_types = {"PERSON", "ORG"}

    def extract(self, sentence: str) -> Dict[str, List[str]]:
        doc = self.nlp(sentence)
        d = sorted([(e.label_.replace("PERSON", "PER"), {"text": e.text, "span": (e.start, e.end - 1)}) for e in doc.ents if e.label_ in self.valid_entity_types], key=lambda t: t[0])
        d = {k: list(map(lambda t:t[1], g)) for k, g in groupby(d,  key=lambda t: t[0])}
        return d


class FlairExtractor(Extractor):

    def __init__(self):
        self.nlp = SequenceTagger.load('ner')
        self.valid_entity_types = {"ORG", "PER", "MISC"}

    def extract(self, sentence: str) -> Dict[str, List[str]]:
        doc = Sentence(sentence)
        self.nlp.predict(doc)
        d = sorted([(e.tag, {"text": e.text.strip('?!. '), "span": (e.tokens[0].idx - 1, e.tokens[-1].idx - 1)}) for e in doc.get_spans('ner') if e.tag in self.valid_entity_types],
                     key=lambda t: t[0])
        d = {k: list(map(lambda t:t[1], g)) for k, g in groupby(d,  key=lambda t: t[0])}
        return d


def process_data(path):
    i2sent = {}
    i2relations = defaultdict(list)
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            idx, arg0, relation, arg1, sentence = line.split("\t")
            i2sent[idx] = sentence
            i2relations[idx].append((arg0, relation, arg1))
    return i2sent, i2relations

i2sent, i2relations = process_data('data/DEV.annotations.tsv')

spacy_extractor = SpacyExtractor()
flair_extractor = FlairExtractor()

work_for = 'Work_For'
def get_relevant_ents(relations) -> Dict[str, set]:
    rel_ent = defaultdict(set)
    for cand_per, relation, cand_org in relations:
        if relation != work_for:
            continue
        rel_ent['PER'].add(cand_per)
        rel_ent['ORG'].add(cand_org)
    return rel_ent


total = 0
spacy_score = 0
flair_score = 0
combined_score = 0
combined_real_score = 0
combined_real_score_danit = 0
for i, sent in tqdm(i2sent.items()):
    relations = i2relations[i]
    rel_ents = get_relevant_ents(relations)
    if len(rel_ents['ORG']) == 0 or len(rel_ents['PER']) == 0:
        continue
    sent = sent.strip("\n ()")
    spacy_pred = spacy_extractor.extract(sent)
    flair_pred = flair_extractor.extract(sent)

    for ent_type, ents in rel_ents.items():
        s = spacy_pred[ent_type] if ent_type in spacy_pred else []
        f = flair_pred[ent_type] if ent_type in flair_pred else []
        for ent in ents:
            total += 1
            spacy_score += 1 if ent in s else 0
            flair_score += 1 if ent in f else 0

            combined_score += 1 if ent in f or ent in s else 0
            is_combined_real_score = False
            is_danit = False
            if ent not in f and ent in s:
                is_combined_real_score = True
            if ent not in f and any([ent in i or i in ent for i in s]):
                if 'MISC' in flair_pred and ent in flair_pred['MISC']:
                    is_danit = True

            combined_real_score += 1 if ent in f or ent in s or is_combined_real_score else 0
            combined_real_score_danit += 1 if ent in f or ent in s or is_danit else 0







print(f"flair score {(flair_score/total) *100}%")
print(f"spacy score {(spacy_score/total) *100}%")
print(f"combined score {(combined_score/total) *100}%")
print(f"combined real score {(combined_real_score/total) *100}%")
print(f"combined real score {(combined_real_score_danit/total) *100}%")







