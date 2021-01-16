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
    total_gold = sum([len(v) for v in gold_relation.values()])
    total_pred = sum([len(v) for v in pred_relation.values()])
    correct = 0
    precision_mistakes = []
    recall_mistakes = []
    for sent_id, pred_relation_list in pred_relation.items():
        for (per_pred, org_pred, sentence) in pred_relation_list:
            suc = False
            if sent_id in gold_relation:
                pred_wf_relation = gold_relation[sent_id]
                for (per_gold, org_gold, _) in pred_wf_relation:
                    if per_gold == per_pred and org_gold == org_pred:
                        suc = True
                        correct += 1
            if not suc:
                precision_mistakes.append((sent_id, per_pred, org_pred, sentence))

    for sent_id, pred_relation_list in gold_relation.items():
        for (per_pred, org_pred, sentence) in pred_relation_list:
            suc = False
            if sent_id in pred_relation:
                pred_wf_relation = pred_relation[sent_id]
                for (per_gold, org_gold, _) in pred_wf_relation:
                    if per_gold == per_pred and org_gold == org_pred:
                        suc = True
            if not suc:
                recall_mistakes.append((sent_id, per_pred, org_pred, sentence))

    # error_analysis_recall(recall_mistakes)


    precision = round((correct / total_pred) * 100, 2)
    recall = round((correct / total_gold) * 100, 2)
    f1 = round((2 * precision * recall) / (precision + recall), 2)
    print(f"precision: {precision}%")
    print(f"recall: {recall}%")
    print(f"f1: {f1}%")







