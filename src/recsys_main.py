import pandas as pd
import argparse
import csv
from lenskit import crossfold as xf
from lenskit.datasets import ML100K

# Import own packages
from RecommenderClass import RecommenderSystem
import conf 
from transform_data_representation import transform_sparse_to_dense_data
from compare_original_synthetic_data import DataComparison
from load_synthetic_data_recombine_partition import CombinePartitionedSyntheticDataLoader


def create_save_train_val_test_rec_data(dense_data, fn):
    # For now, no cross-validation, just split the data into 1 train and 1 test set.
    for i, tp in enumerate(xf.partition_users(data=dense_data, partitions=1, method=xf.SampleN(5), rng_spec=1)):
        train = tp.train
        test = tp.test
        test.to_csv(f'{conf.SYN_DATA_DIR}syn_test_{fn}.csv')
        print("[INFO] Train/test split created")

    for i, tp in enumerate(xf.partition_users(data=train, partitions=1, method=xf.SampleN(5), rng_spec=1)):
        train = tp.train
        val = tp.test
        train.to_csv(f'{conf.SYN_DATA_DIR}syn_train_{fn}.csv')
        val.to_csv(f'{conf.SYN_DATA_DIR}syn_val_{fn}.csv')
        print("[INFO] Train/val split created")

    return train, val, test


def model_tuning(train, val, test, a):
    if (a == "itemKNN"):
        ## Import is this way needed due to otherwise rise of error when tuning/testing different algorithms.
        import tensorflow as tf

        # Set parameter ranges to try
        params = {
            'nnbrs': [20, 40, 50, 100],
            'aggregate' : 'sum',
            'center' : False
        }

        highest_res = 0
        best_param = []
        # GridSearch manually the best hyperparameter settings
        for nbrs in params['nnbrs']:
            aggregate=params['aggregate']
            center=params['center']
            recsys = RecommenderSystem(train, val)

            # Fit the model to the training/validation data
            recs =  recsys.itemKNN(nbrs, aggregate, center)

            # Analyze performance and keep track of best performance
            res = recsys.analyze_performance(recs)
            res_float = float(res.mean().ndcg)
            if (res_float > highest_res):
                best_param = nbrs
                highest_res = res_float
        print(f'final best param: {best_param} - result = {highest_res}')

        # Compute test score
        recsys = RecommenderSystem(train, test)
        recs =  recsys.itemKNN(best_param, params['aggregate'], params['center'])
        res = recsys.analyze_performance(recs)
        res_float = float(res.mean().ndcg)
        print(f'Final test result = {res_float}')

    elif (a == "userKNN"):
        ## Import is this way needed due to otherwise rise of error when tuning/testing different algorithms.
        import tensorflow as tf

        # Set parameter ranges to try
        params = {
            'nnbrs' : [20, 40, 50, 100],
            'aggregate' : 'sum',
            'center' : False
        }
        recsys = RecommenderSystem(train, val)
        
        highest_res = 0
        best_param = []
        # GridSearch manually the best hyperparameter settings
        for nbrs in params['nnbrs']:
            aggregate=params['aggregate']
            center=params['center']
            recsys = RecommenderSystem(train, val)

            # Fit the model to the training/validation data
            recs =  recsys.userKNN(nbrs, aggregate, center)

            # Analyze performance and keep track of best performance
            res = recsys.analyze_performance(recs)
            res_float = float(res.mean().ndcg)
            if (res_float > highest_res):
                best_param = nbrs
                highest_res = res_float
        print(f'final best param: {best_param} - result = {highest_res}')

        # Compute test score
        recsys = RecommenderSystem(train, test)
        recs =  recsys.userKNN(best_param, params['aggregate'], params['center'])
        res = recsys.analyze_performance(recs)
        res_float = float(res.mean().ndcg)
        print(f'Final test result = {res_float}')

    elif (a == "BPRMF"):
        # Set parameter ranges to try
        params = {
            'features' : [20, 40, 50, 60, 80],
            'epochs' : [30,50,70],
            'batch_size' : [350,500,650]
        }
        
        # GridSearch manually the best hyperparameter settings
        highest_res = 0
        best_params = []
        for f in params['features']:
            for e in params['epochs']:
                for b in params['batch_size']: 
                    recsys = RecommenderSystem(train, val)

                    # Fit the model to the training/validation data
                    recs =  recsys.BPRMF(f, e, b)

                     # Analyze performance and keep track of best performance
                    res = recsys.analyze_performance(recs)
                    res_float = float(res.mean().ndcg)
                    if (res_float > highest_res):
                        best_params = [f, e, b]
                        highest_res = res_float
        print(f'final best params: {best_params} - result = {highest_res}')

        # Compute test score
        recsys = RecommenderSystem(train, test)
        recs = recsys.BPRMF(best_params[0], best_params[1], best_params[2])
        res = recsys.analyze_performance(recs)
        res_float = float(res.mean().ndcg)
        print(f'Final test result = {res_float}')

    else:
        print("[INFO] Please provide correct algorithm.")
    
    # Compute metrics
    metrics = recsys.analyze_performance(recs)
    print(f"Computed ndcg metrics:\n{metrics}")


def main(i, a, o, to_split):
    # Load and save the synthetic data (split in train & test set)
    if to_split == "True":
        """
        Input data will be split into training, validation, and testing set.
        """
        if i == 'ml100k':
            ml100k = ML100K('ml-100k')
            dense_data = ml100k.ratings
            dense_data = dense_data[['user', 'item', 'rating']]
        else:
            input_path = f"{conf.SYN_DATA_DIR}{i}.csv"
            input_data = pd.read_csv(input_path, sep=',', encoding="latin-1").fillna(0)
            dense_data = transform_sparse_to_dense_data(input_data)
            print("[INFO] Transformed sparse data to dense representation")
        syn_train, syn_val, syn_test = create_save_train_val_test_rec_data(dense_data, o)
    else:
        """
        Training, validation, and testing set will be loaded.
        """
        syn_train_path = f"{conf.SYN_DATA_DIR}syn_train_{i}.csv" 
        syn_val_path = f"{conf.SYN_DATA_DIR}syn_val_{i}.csv"
        syn_test_path = f"{conf.SYN_DATA_DIR}syn_test_{i}.csv"

        syn_train = pd.read_csv(syn_train_path, sep=',', encoding="latin-1").fillna(0)[['user', 'item', 'rating']]
        syn_val = pd.read_csv(syn_val_path, sep=',', encoding="latin-1").fillna(0)[['user', 'item', 'rating']]
        syn_test = pd.read_csv(syn_test_path, sep=',', encoding="latin-1").fillna(0)[['user', 'item', 'rating']]

    # Apply the recommender system algorithm to the original and new data
    model_tuning(syn_train, syn_val, syn_test, a)
 

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-D", "--data", type=str, help="Which sparse input data to use for the recommendation", default="750eps_non_partitioned")
    ap.add_argument("-S", "--split", type=str, help="If input data needs to be split or not", default="True")
    ap.add_argument("-A", "--algo", type=str, help="Which recommender system algorithm to use")
    ap.add_argument("-O", "--output", type=str, help="added string to train/test output csv files", default="750eps_non_partitioned")
    args = vars(ap.parse_args())
   
    main(args['data'], args['algo'], args['output'], args['split'])
