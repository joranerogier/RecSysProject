import pandas as pd
import argparse
import csv
from lenskit import crossfold as xf

# Import own packages
from RecommenderClass import RecommenderSystem
import conf 
from sparse_to_dense import transform_sparse_to_dense_data
from compare_original_synthetic_data import DataComparison
from load_synthetic_data_recombine_partition import CombinePartitionedSyntheticDataLoader
from dense_to_sparse import transform_dense_to_sparse_data

def create_save_train_test_rec_data(dense_data, fn):
    # For now, no cross-validation, just split the data into 1 train and 1 test set.
    for i, tp in enumerate(xf.partition_users(data=dense_data, partitions=1, method=xf.SampleN(5), rng_spec=1)):
        train = tp.train
        test = tp.test
        train.to_csv(f'{conf.SYN_DATA_DIR}syn_train_{fn}.csv')
        test.to_csv(f'{conf.SYN_DATA_DIR}syn_test_{fn}.csv')
        print("[INFO] Train/test split created")
        print(test)

    # Convert dense data to sparse again, since the recommender system algorithms require sparse data
    train_sparse = transform_sparse_to_dense_data(train)
    test_sparse = transform_sparse_to_dense_data(test)
    return train_sparse, test_sparse


def main(sparse_data, a, o):
    # Load and save the synthetic data (split in train & test set)
    dense_data = transform_sparse_to_dense_data(sparse_data)
    print("[INFO] Transformed sparse data to dense representation")
    syn_train, syn_test = create_save_train_test_rec_data(dense_data, o)
    
    # Apply the recommender system algorithm to the original and new data
    recsys = RecommenderSystem(syn_train, syn_test)
    
    if (a == "itemKNN"):
        recs =  recsys.itemKNN()
        print(f"Evaluation itemKNN: {recs}")
    elif (a == "userKNN"):
        recs = recsys.userKNN()
        print(f"Evaluation userKNN: {recs}")
    elif (a == "BPRMF"):
        recs =  recsys.BPRMF()
        print("BPRMF not yet implemented")
    else:
        print("Provide correct algorithm")
    
    # Compute metrics
    metrics = recsys.analyze_performance(recs)
    print(f"Computed ndcg metrics:\n{metrics}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-D", "--data", type=str, help="Which sparse input data to use for the recommendation")
    ap.add_argument("-A",  "--algo", type=str, help="Which recommender system algorithm to use")
    ap.add_argument("-O", "--output", type=str, help="added string to train/test output csv files", default="test")
    args = vars(ap.parse_args())

    input_path = f"{conf.SYN_DATA_DIR}{args['data']}.csv"
    input_data = pd.read_csv(input_path, sep=',', encoding="latin-1").fillna("")

    recsys_algo = args['algo']
    
    main(input_data, recsys_algo, args['output'])