import en_core_web_md
from itertools import product
from collections import defaultdict
from spacy import displacy
from extract import ProcessAnnotatedData, RelationSentence, load_from_pickle, TRAIN_F, TEST_F

# nlp = en_core_web_md.load()
# sentence = "Also being considered are Judge Ralph K. Winter of the 2nd U.S. Circuit Court of Appeals in New York City and Judge Kenneth Starr of the U.S. Circuit Court of Appeals for the District of Columbia , said the source , who spoke on condition of anonymity ."
# analyze = nlp(sentence)
# for np in analyze.noun_chunks:
#     print(np.text, np.root.text, np.root.dep_, np.root.head.text)
# for ne in analyze.ents:
#     print(ne.text, ne.root.ent_type_, ne.root.text, ne.root.dep_, ne.root.head.text)
# displacy.serve(analyze, style='dep')
# print()


def is_person_token(token, person):
    if person["span"][1] <= token.i <= person["span"][1]:
        return True
    return False


def get_entity_head(doc, entity):
    for j in range(entity["span"][0], entity["span"][1] + 1):
        if doc[j].head.i > entity["span"][1] or doc[j].head.i < entity["span"][0]:
            return doc[j]
    return None


def is_work_for(org_head_token, person):
    if org_head_token.dep_ == "pobj" and org_head_token.head.dep_ == "prep":
        if is_person_token(org_head_token.head.head, person):
            return True
        elif org_head_token.head.head.dep_ == "appos" and is_person_token(org_head_token.head.head.head, person):
            return True
    elif org_head_token.dep_ in ["nmod", "compound"] and org_head_token.head.dep_ == "appos" and is_person_token(org_head_token.head.head, person):
        return True

    return False

data = load_from_pickle(TRAIN_F)
all_preds = defaultdict(list)
count_preds = 0
for rel_sent in data.i2sentence.values():
    for (cand_per, cand_org) in product(rel_sent.entities['PER'], rel_sent.entities['ORG']):
        org_head_token = get_entity_head(rel_sent.analyzed, cand_org)
        if not org_head_token:
            print(f"can't find org head: {cand_org}")
            continue
        if is_work_for(org_head_token, cand_per):
            all_preds[rel_sent.idx].append((cand_per["text"], "Work_For", cand_org["text"]))
            count_preds += 1

correct = 0
fuzzy_correct = 0
wrong_pred = []
for idx, preds in all_preds.items():
    gold = data.i2relations[idx]
    for pred_rel in preds:
        added = False
        for gold_rel in gold:
            if gold_rel == pred_rel:
                correct += 1
                added = True
                break
            elif (gold_rel[0] in pred_rel[0] or pred_rel[0] in gold_rel[0]) and \
                    (gold_rel[2] in pred_rel[2] or pred_rel[2] in gold_rel[2]) and \
                    pred_rel[1] == gold_rel[1]:
                fuzzy_correct += 1
                added = True
                break
        if not added:
            wrong_pred.append((idx, pred_rel[0], pred_rel[2]))
print(f"Number of predictions: {count_preds}")
print(f"Exact match: {correct}")
print(f"Partial match: {fuzzy_correct}")
print(f"Wrong predictions number: {len(wrong_pred)}")
print("Wrong predictions:")
for wrong in wrong_pred:
    print(wrong)

# TRAIN:
# Number of predictions: 48
# Exact match: 39
# Partial match: 1
# Wrong predictions number: 8
# Wrong predictions:
# ('sent52', 'Kenneth Starr', 'District of Columbia')
# ('sent493', 'Toth', 'Legislature')
# ('sent541', 'Albert O. Harjula', 'Thomaston')
# ('sent541', 'Paul Fournier', 'the state Department of Inland Fisheries')
# ('sent670', 'Darryl Breniser', 'Blue Ball')
# ('sent674', 'shearer', 'Wool Wizards')
# ('sent4628', 'Gerald R. Ford', 'Warren Commission')
# ('sent4650', 'Tidwell', 'CIA')
#
# TEST:
# Number of predictions: 26
# Exact match: 16
# Partial match: 6
# Wrong predictions number: 4
# Wrong predictions:
# ('sent903', 'James Francis', 'Dayton')
# ('sent1702', 'Leonard Lee', 'Chinese Times')
# ('sent1787', 'Dave Olson', 'the Payette National Forest')
# ('sent4708', 'Surratt', 'Confederate')