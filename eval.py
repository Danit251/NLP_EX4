import sys
# from extract import ProcessAnnotatedData
from collections import defaultdict

from tqdm import tqdm


def process_data(path):
    i2relations = defaultdict(list)
    with open(path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            idx, per, relation, org, sentence = line.split("\t")
            if relation == "Work_For":
                i2relations[idx].append((per, org, sentence))
    return i2relations


if __name__ == '__main__':
    gold_relation = process_data(sys.argv[1])
    pred_relation = process_data(sys.argv[2])
    total_pred = sum([len(v) for v in pred_relation.values()])
    total_gold = sum([len(v) for v in gold_relation.values()])
    correct = 0
    examples_wrong = []

    for sent_id, relation_list in pred_relation.items():
        for (per_pred, org_pred, sentence) in relation_list:
            suc = False
            if sent_id in gold_relation:
                gold_wf_relation = gold_relation[sent_id]
                for (per_gold, org_gold, _) in relation_list:
                    if per_gold == per_pred and org_gold == org_pred:
                        suc = True
                        correct += 1
            if not suc:
                examples_wrong.append((per_pred, org_pred, sentence))

    precision = correct / total_pred
    recall = correct / total_gold
    f1 = (2 * precision * recall) / (precision + recall)
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")







