import pandas as pd
import argparse
import csv
from lenskit import crossfold as xf

# Import own packages
from RecommenderClass import RecommenderSystem
import conf 
from transform_data_representation import transform_sparse_to_dense_data, transform_dense_to_sparse_data
from compare_original_synthetic_data import DataComparison
from load_synthetic_data_recombine_partition import CombinePartitionedSyntheticDataLoader


def create_save_train_test_rec_data(dense_data, fn):
    # For now, no cross-validation, just split the data into 1 train and 1 test set.
    for i, tp in enumerate(xf.partition_users(data=dense_data, partitions=1, method=xf.SampleN(5), rng_spec=1)):
        train = tp.train
        test = tp.test
        train.to_csv(f'{conf.SYN_DATA_DIR}syn_train_{fn}.csv')
        test.to_csv(f'{conf.SYN_DATA_DIR}syn_test_{fn}.csv')
        print("[INFO] Train/test split created")

    # Convert dense data to sparse again, since the recommender system algorithms require sparse data
    train_sparse = transform_dense_to_sparse_data(train)
    test_sparse = transform_dense_to_sparse_data(test)
    train_sparse.to_csv(f'{conf.SYN_DATA_DIR}syn_train_sparse_{fn}.csv')
    test_sparse.to_csv(f'{conf.SYN_DATA_DIR}syn_test_sparse_{fn}.csv')
    return train_sparse, test_sparse


def main(input_data, a, o, to_split):
    # Load and save the synthetic data (split in train & test set)
    if to_split == "True":
        input_path = f"{conf.SYN_DATA_DIR}{input_data}.csv"
        input_data = pd.read_csv(input_path, sep=',', encoding="latin-1").fillna(0)
        dense_data = transform_sparse_to_dense_data(input_data)
        print("[INFO] Transformed sparse data to dense representation")
        syn_train, syn_test = create_save_train_test_rec_data(dense_data, o)
    else:
        syn_train_path = f"{conf.SYN_DATA_DIR}syn_train_{input_data}.csv" 
        syn_test_path = f"{conf.SYN_DATA_DIR}syn_test_{input_data}.csv"
        syn_train = pd.read_csv(syn_train_path, sep=',', encoding="latin-1").fillna(0)
        syn_test = pd.read_csv(syn_test_path, sep=',', encoding="latin-1").fillna(0)
        syn_train = syn_train[['user', 'item', 'rating']]
        syn_test = syn_test[['user', 'item', 'rating']]

    
    # Apply the recommender system algorithm to the original and new data
    recsys = RecommenderSystem(syn_train, syn_test)
    print(syn_train)
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
    ap.add_argument("-S", "--split", type=str, help="If input data needs to be split or not", default=True)
    ap.add_argument("-A", "--algo", type=str, help="Which recommender system algorithm to use")
    ap.add_argument("-O", "--output", type=str, help="added string to train/test output csv files", default="test")
    args = vars(ap.parse_args())

    recsys_algo = args['algo']
    
    main(args['data'], recsys_algo, args['output'], args['split'])
