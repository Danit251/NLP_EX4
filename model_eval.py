from extract import load_from_pickle, predict, select_features, write_results, TRAIN_F, TEST_F, RelationsVectorizer, XGBClassifier
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier


train = load_from_pickle(TRAIN_F)
test = load_from_pickle(TEST_F)

vectorizer = RelationsVectorizer(train, test)
# save_to_pickle(model, f"models/{model_name}.pkl")
for ent_n in [50, 100, 200, 1000]:
    for xgb in ["XGB", "RF"]:
        if xgb == "XGB":
            model = XGBClassifier(n_estimators=ent_n)
        else:
            model = RandomForestClassifier(n_estimators=ent_n)

        model.fit(vectorizer.train_vectors, vectorizer.train_labels)

        predicted_labels = predict(model, vectorizer.test_vectors)

        write_results(f"PRED.annotations_{str(ent_n)}_{xgb}.txt", test.op_relations, predicted_labels)

        ranked_features = select_features(model, vectorizer.train_vectors, vectorizer.train_labels,
                                          vectorizer.dv.feature_names_)
        print(ranked_features)

        report = classification_report(vectorizer.test_labels, predicted_labels)
        print(report)
        mcc = matthews_corrcoef(vectorizer.test_labels, predicted_labels)
        print(f"MCC: {mcc}")
        with open(f"models/report_{xgb}_{str(ent_n)}", "w") as f:
            f.write(f"model name: {xgb}_{str(ent_n)}\n")
            f.write(report)
            f.write("\n~~~~~~~~~~\n")
            f.write(f"MCC: {mcc}")
            f.write("\n~~~~~~~~~~\n")
            f.write(f"Rank: {ranked_features}")