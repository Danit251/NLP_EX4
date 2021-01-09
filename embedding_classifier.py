import functools

import numpy as np
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, matthews_corrcoef
from xgboost import XGBClassifier
from extract import load_from_pickle, ProcessAnnotatedData, save_to_pickle, TRAIN_F, TEST_F
import en_vectors_web_lg
LOAD_FROM_PICKLE = True

class WeVectorizer:
    def __init__(self,  train_data, test_data):
        self.vectorizer = en_vectors_web_lg.load()
        self.train_vec = self.vectorizer_data(train_data.op_relations)
        self.train_labels = train_data.labels

        self.test_vec = self.vectorizer_data(test_data.op_relations)
        self.test_labels = test_data.labels

    def vectorizer_data(self, relations):
        vecs = []
        for sent_id, per_cand, org_cand, sent_raw  in relations:
            sent = sent_raw.strip("().\n")
            org = org_cand['text']
            per = per_cand['text']
            sent_clean = sent.replace(org, "").replace(per, "")
            vecs.append(self.vec_sent(sent_clean, per, org))
        vecs = np.array(vecs)
        return vecs

    def vec_sent(self, sent, per_candidate, org_candidate):
            toks = [t for t in self.vectorizer(sent) if not any([t.is_space, t.is_punct, t.is_stop, t.is_currency]) and t.has_vector]
            sent_vecs = np.array([t.vector for t in toks]).mean(axis=0)
            per_vec = self.vectorize_ent(per_candidate)
            org_vec = self.vectorize_ent(org_candidate)
            res = np.concatenate([sent_vecs, per_vec, org_vec])
            return res


    def vectorize_ent(self, org_candidate):
        return np.array([t.vector for t in self.vectorizer(org_candidate)]).mean(axis=0)


def main():
    # model_name = "glove_300_xgboost_sub_sample_defualt_ne_50_sent"
    model_name = "glove_300_sgd_sent_per_org_max_iter_1000"
    np.random.seed(42)
    if LOAD_FROM_PICKLE:
        train = load_from_pickle(TRAIN_F)
        test = load_from_pickle(TEST_F)
    else:
        train = ProcessAnnotatedData(sys.argv[1])
        save_to_pickle(train, TRAIN_F)
        test = ProcessAnnotatedData(sys.argv[2])
        save_to_pickle(test, TEST_F)

    vectorizer = WeVectorizer(train, test)
    # model = XGBClassifier(n_estimators=50)
    # model = RandomForestClassifier(n_estimators=20)
    # model = LogisticRegression(max_iter=1000)
    model = SGDClassifier(max_iter=1000)
    model.fit(vectorizer.train_vec, vectorizer.train_labels)
    save_to_pickle(model, f"models/{model_name}.pkl")

    preds = model.predict(vectorizer.test_vec)
    report = classification_report(vectorizer.test_labels, preds)

    print(report)
    mcc = matthews_corrcoef(vectorizer.test_labels, preds)
    print(f"mcc: {mcc}")
    with open(f"models/report_{model_name}.txt", "w") as f:
        f.write(f"mcc: {mcc}\n")
        f.write(report)



if __name__ == '__main__':
    main()
