import en_core_web_md



nlp = en_core_web_md.load()


class Sentence:
    def __init__(self, sentence):
        self.doc = nlp(sentence)
        self.ents = [{'text': ne.text, 'type':ne.root.ent_type_} for ne in self.doc.ents_]
        self.ents_type = [i['type'] for i in self.ents]
        self.is_candidate_live_in  = self.is_candidate_livein()
        self.is_candidate_work_for = self.is_candidate_workfor()


    def is_candidate_workfor(self):
        return 'ORG' in self.ents_type and 'PERSON' in self.ents_type

    def is_candidate_livein(self):
        return ('NORP' in self.ents_type or 'GPE' in self.ents_type) and 'PERSON' in self.ents_type



class EvalSentences:
    def __init__(self, data):






def main():
    pass


if __name__ == '__main__':
    main()
