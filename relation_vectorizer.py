from collections import defaultdict

import en_core_web_md
from flair.data import Sentence
from sklearn.feature_extraction import DictVectorizer
from common import SPAN
import numpy as np
from tqdm import tqdm
from flair.embeddings import TransformerWordEmbeddings

class RelationsVectorizer:

    def __init__(self, i2sentence, op_relations, embedding_type='', dv=None):

        self.features = self.get_features(i2sentence, op_relations)
        self.remove_redundant_features()
        self.normalize_features()

        if not dv:
            self.dv = DictVectorizer()
            self.f_vectors = self.dv.fit_transform(self.features).toarray()
        else:
            self.dv = dv
            self.f_vectors = self.dv.transform(self.features).toarray()
        if embedding_type:
            self.embedding_vectorizer = WeVectorizer(op_relations, embedding_type)
            self.e_vectors = self.embedding_vectorizer.vectors
            self.vectors = self.merge_vectors(self.f_vectors, self.e_vectors)
        else:
            self.vectors = self.f_vectors

    def normalize_features(self):
        sent_max = max([f["dist_sent"] for f in self.features])
        tree_max = max([f["dist_tree"] for f in self.features])
        for f in self.features:
            f["dist_sent"] /= sent_max
            f["dist_tree"] /= tree_max

    def merge_vectors(self, feat_vectors, embedding_vectors):
        merged = []
        for f_vec, e_vec in zip(feat_vectors, embedding_vectors):
            merged.append(np.concatenate([f_vec, e_vec]))
        return np.array(merged)

    def remove_redundant_features(self):
        stat = self.create_features_stat()
        for f_dict in self.features:
            for f_name, f_val in f_dict.items():
                f_stat = stat[f_name][f_val]
                if (f_name == "dist_sent" and (f_val < 4 or f_val > 20)) or \
                        (f_name == "dist_tree" and (f_val < 3 or f_val > 10)):
                    f_dict[f_name] = 0
                if f_name not in ["dist_sent", "dist_tree"] and f_stat < 15:
                    f_dict[f_name] = "OTHER"

    def create_features_stat(self):
        stat = defaultdict(lambda: defaultdict(int))
        for f in self.features:
            for f_name, f_val in f.items():
                stat[f_name][f_val] += 1

        for f_name, f_stat in stat.items():
            stat[f_name] = {k: v for k, v in sorted(f_stat.items(), key=lambda item: item[1], reverse=True)}

        return stat

    def get_features(self, i2sentences, relations):
        features = []
        for relation in relations:
            feature = self.get_relation_features(relation, i2sentences[relation[0]])
            features.append(feature)
        return features

    def get_relation_features(self, relation, sentence):
        features = {}
        for f_feature in [self.f_pre_pos, self.f_after_pos, self.f_distance_in_sentence, self.f_distance_in_tree,
                          self.f_cur_pos]:
            f_feature(features, relation[1], relation[2], sentence)
        return features

    def f_pre_pos(self, features, person, org, sentence):
        person_start_index = person[SPAN][0]
        if person_start_index != 0:
            features["pre_person_pos"] = sentence.analyzed[person_start_index - 1].pos_
        else:
            features["pre_person_pos"] = "START"

        org_start_index = org[SPAN][0]
        if org_start_index != 0:
            features["pre_org_pos"] = sentence.analyzed[org_start_index - 1].pos_
        else:
            features["pre_org_pos"] = "START"

        return features

    def f_cur_pos(self, features, person, org, sentence):
        person_start_index = person[SPAN][0]
        features["cur_person_pos"] = sentence.analyzed[person_start_index].pos_

        org_start_index = org[SPAN][0]
        features["cur_org_pos"] = sentence.analyzed[org_start_index].pos_
        return features

    def f_after_pos(self, features, person, org, sentence):
        person_end_index = person[SPAN][1]
        try:
            if person_end_index != len(sentence.analyzed.doc) - 1:
                features["after_person_pos"] = sentence.analyzed[person_end_index + 1].pos_
            else:
                features["after_person_pos"] = "END"
        except Exception as error:
            pass

        org_end_index = org[SPAN][1]
        if org_end_index != len(sentence.analyzed.doc) - 1:
            features["after_org_pos"] = sentence.analyzed[org_end_index + 1].pos_
        else:
            features["after_org_pos"] = "END"

    def f_distance_in_sentence(self, features, person, org, sentence):
        person_start_index = person[SPAN][0]
        org_start_index = org[SPAN][0]
        features["dist_sent"] = abs(person_start_index - org_start_index)

    def f_distance_in_tree(self, features, person, org, sentence):
        person_start_index = person[SPAN][0]
        person_end_index = person[SPAN][1]
        org_start_index = org[SPAN][0]
        cur_head = sentence.analyzed[org_start_index]
        dist = 1
        # all_deps = defaultdict(int)
        while cur_head.dep_ != "ROOT" and (person_start_index > cur_head.i or cur_head.i > person_end_index):
            dist += 1
            cur_head = cur_head.head
        #     all_deps[f"dep_{cur_head.dep_}"] += 1
        #
        # if cur_head.dep_ != "ROOT":
        #     all_deps[f"dep_{cur_head.dep_}"] += 1
        #     features.update(all_deps)

        if cur_head.dep_ == "ROOT":
            features["dist_tree"] = 0
        else:
            features["dist_tree"] = dist



class WeVectorizer:

    def __init__(self,  op_relations, vectorizer='spacy'):
        if vectorizer == 'spacy':
            self.vectorizer = en_core_web_md.load()
        else:
            self.vectorizer = TransformerWordEmbeddings('roberta-base')
        self.vectors = self.vectorizer_data(op_relations)

    def _vectorizer_data(self, relations):
        vecs = []
        for sent_id, per_cand, org_cand, sent_raw  in tqdm(relations):
            sent = sent_raw.strip("().\n")
            org = org_cand['text']
            per = per_cand['text']
            sent_clean = sent.replace(org, "").replace(per, "")
            vecs.append(self.vec_sent(sent_clean, per, org))
        vecs = np.array(vecs)
        return vecs

    def vectorizer_data(self, relations):
        vecs = []
        for sent_id, per_cand, org_cand, sent_raw in tqdm(relations):
            sent = sent_raw.strip("().\n")
            sent = Sentence(sent)
            self.vectorizer.embed(sent)
            vecs.append(sent[0].embedding.cpu().detach().numpy())
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
