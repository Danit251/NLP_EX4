from abc import ABC, abstractmethod
from itertools import groupby
from typing import Dict, Union, Tuple

import en_core_web_md
from flair.data import Sentence
from flair.models import SequenceTagger


class Extractor(ABC):

    @abstractmethod
    def extract(self, sentence: str) ->Dict[str, Dict[str, Union[str, Tuple]]]:
        pass


class SpacyExtractor(Extractor):

    def __init__(self):
        self.nlp = en_core_web_md.load()
        self.valid_entity_types = {"PERSON", "ORG"}

    def extract(self, sentence: str) -> Dict[str, Dict[str, Union[str, Tuple]]]:
        doc = self.nlp(sentence)
        d = sorted([(e.label_.replace("PERSON", "PER"), {"text": e.text, "span": (e.start, e.end - 1)}) for e in doc.ents if e.label_ in self.valid_entity_types], key=lambda t: t[0])
        d = {k: list(map(lambda t: t[1], g)) for k, g in groupby(d,  key=lambda t: t[0])}
        return d


class FlairExtractor(Extractor):

    def __init__(self):
        self.nlp = SequenceTagger.load('ner')
        self.valid_entity_types = {"ORG", "PER", "MISC"}

    def extract(self, sentence: str) -> Dict[str, Dict[str, Union[str, Tuple]]]:
        doc = Sentence(sentence)
        self.nlp.predict(doc)
        d = sorted([(e.tag, {"text": e.text, "span": (e.tokens[0].start_pos, len(e.tokens))}) for e in doc.get_spans('ner') if e.tag in self.valid_entity_types],
                     key=lambda t: t[0])
        d = {k: list(map(lambda t: t[1], g)) for k, g in groupby(d,  key=lambda t: t[0])}
        return d


