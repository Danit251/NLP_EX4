RELATION = "Work_For"
PERSON = "PER"
ORG = "ORG"
TEXT = "text"
TYPE = "type"
SPAN = "span"


def write_results(f_name, op_relations, predicted_labels):
    rel_set = set()
    with open(f_name, "w") as f_res:
        for i, (idx, person, org, sentence) in enumerate(op_relations):
            rel_str = "_".join([idx, person[TEXT], RELATION, org[TEXT]])
            if predicted_labels[i] == "1" and rel_str not in rel_set:
                f_res.write("\t".join([idx, person[TEXT], RELATION, org[TEXT], f"( {sentence} )\n"]))
                rel_set.add(rel_str)