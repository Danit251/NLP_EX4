import sys
import en_core_web_md
from collections import defaultdict
import numpy as np

nlp = en_core_web_md.load()
RELATION = "Work_For"
PERSON = "PERSON"
ORG = "ORG"


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


def create_data(path):
    i2sent, i2relations = process_data(path)
    with open(f"data_{RELATION}.txt", mode="w") as output_f:
        for idx, sentence in i2sent.items():
            examples = get_relevant_examples(sentence, i2relations[idx])
            for example in examples:
                output_f.write("\t".join(example) + f"\t{idx}\t{sentence}")


def get_relevant_examples(sentence, relations):
    entities = nlp(sentence).entities
    persons = list(set([ne.text for ne in entities if ne.root.ent_type_ == PERSON]))
    orgs = list(set([ne.text for ne in entities if ne.root.ent_type_ == ORG]))
    examples = []

    for person in persons:
        for org in orgs:
            if is_example_relevant(person, org, relations):
                examples.append(("1", person, org))
            else:
                examples.append(("0", person, org))

    return examples


def is_example_relevant(person, org, relations):
    for arg0, rel, arg1 in relations:
        if rel == RELATION and (person in arg0 or arg0 in person) and (org in arg1 or arg1 in org):
            return True
    return False


def create_vectors(sentence, entities):
    return


def get_vector(doc, sentence):
    vector = np.array()
    return vector


def main():
    create_data(sys.argv[1])


if __name__ == '__main__':
    main()
