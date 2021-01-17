import sys

from data_processor import ProcessAnnotatedData, RelationSentence
from relation_vectorizer import RelationsVectorizer
from pipeline import RelationExtractionPipeLine
TRAIN_F = "train_data.pkl"
TEST_F = "test_data.pkl"

model_name = "model_XGB_1000_feat_deps"



def dev_main():
    use_cache = True
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    pipeline = RelationExtractionPipeLine()
    pipeline.run_train_pipeline(train_path, test_path, use_cache)



def main():
    test_path = sys.argv[1]
    output_path = sys.argv[2]
    if not test_path.endswith('.txt'):
        print("Please use a txt  file.")
        exit(1)
    train_mode = True
    pipeline = RelationExtractionPipeLine(train_mode=True)
    # pipeline = RelationExtractionPipeLine(test_path, train_path, use_cache)
    pipeline.run()






if __name__ == '__main__':
    # main()
    dev_main()
