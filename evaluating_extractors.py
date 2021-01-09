from collections import defaultdict
from typing import Dict
from tqdm import tqdm

from extractors import SpacyExtractor, FlairExtractor

work_for = 'Work_For'

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



def get_relevant_ents(relations) -> Dict[str, set]:
    rel_ent = defaultdict(set)
    for cand_per, relation, cand_org in relations:
        if relation != work_for:
            continue
        rel_ent['PER'].add(cand_per)
        rel_ent['ORG'].add(cand_org)
    return rel_ent


def run_evaluation():
    spacy_extractor = SpacyExtractor()
    flair_extractor = FlairExtractor()
    i2sent, i2relations = process_data('data/TRAIN.annotations.tsv')

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
            s = set([i['text'] for i in s])
            f = flair_pred[ent_type] if ent_type in flair_pred else []
            f = set([i['text'] for i in f])
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
    print(f"flair score {(flair_score / total) * 100}%")
    print(f"spacy score {(spacy_score / total) * 100}%")
    print(f"combined score {(combined_score / total) * 100}%")
    print(f"combined real score {(combined_real_score / total) * 100}%")
    print(f"combined real score {(combined_real_score_danit / total) * 100}%")


# if __name__ == '__main__':
#     run_evaluation()







