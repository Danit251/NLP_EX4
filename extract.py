import sys

from data_processor import ProcessAnnotatedData, RelationSentence
from relation_vectorizer import RelationsVectorizer
from pipeline import RePipeLine
TRAIN_F = "train_data.pkl"
TEST_F = "test_data.pkl"

model_name = "model_XGB_1000_feat_deps"




def main():
    use_cache = True  #TODO renove before delivery
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    pipeline = RePipeLine(train_path, test_path, use_cache)
    pipeline.run()






if __name__ == '__main__':
    main()
