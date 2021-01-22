import sys
from pipeline import RelationExtractionPipeLine


def dev_main():
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    pipeline = RelationExtractionPipeLine()
    pipeline.run_train_pipeline(train_path, test_path, use_cache=True, model_grid_search=False)


def main():
    test_path = sys.argv[1]
    output_path = sys.argv[2]
    if not test_path.endswith('.txt'):
        print("Please use a txt  file.")
        exit(1)
    pipeline = RelationExtractionPipeLine()
    pipeline.run(test_path, output_path)






if __name__ == '__main__':
    dev_main()
    main()
