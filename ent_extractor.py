from extractors import SpacyExtractor, FlairExtractor
from collections import defaultdict
from common import PERSON, ORG, TEXT, SPAN


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

    def extract(self, analyzed_sent):
        spacy_entities = self.ext_spacy.extract(analyzed_sent.text)
        flair_entities = self.ext_flair.extract(analyzed_sent.text)
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

