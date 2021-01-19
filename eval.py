import sys
# from extract import ProcessAnnotatedData
from collections import defaultdict

from tqdm import tqdm


def main(gold_path, pred_path):
    gold_relation = process_data(gold_path)
    pred_relation = process_data(pred_path)
    evaluate_result(gold_relation, pred_relation)


def process_data(path):
    i2relations = defaultdict(list)
    with open(path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            idx, per, relation, org, sentence = line.split("\t")
            if relation == "Work_For":
                i2relations[idx].append((per, org, sentence))
    return i2relations


def evaluate_result(gold_relation, pred_relation):
    total_gold = sum([len(v) for v in gold_relation.values()])
    total_pred = sum([len(v) for v in pred_relation.values()])
    correct = 0
    precision_mistakes = []
    recall_mistakes = []
    correct_gen = 0
    examples_wrong = []
    for sent_id, pred_relation_list in pred_relation.items():
        for (per_pred, org_pred, sentence) in pred_relation_list:
            suc = False
            if sent_id in gold_relation:
                for (per_gold, org_gold, _) in gold_relation[sent_id]:
                    if per_gold == per_pred and org_gold == org_pred:
                        suc = True
                        correct += 1
                        correct_gen += 1
                    elif (per_gold in per_pred or per_pred in per_gold) and (
                            org_gold in org_pred or org_pred in org_gold):
                        correct_gen += 1
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
    print("Regular accuracy:")
    print(f"precision: {precision}%")
    print(f"recall: {recall}%")
    print(f"f1: {f1}%")
    print()
    precision_gen = round((correct_gen / total_pred) * 100, 2)
    recall_gen = round((correct_gen / total_gold) * 100, 2)
    f1_gen = round((2 * precision_gen * recall_gen) / (precision_gen + recall_gen), 2)
    print("Generalized accuracy:")
    print(f"precision: {precision_gen}%")
    print(f"recall: {recall_gen}%")
    print(f"f1: {f1_gen}%")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])






