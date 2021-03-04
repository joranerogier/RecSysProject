import pandas as pd
from sdv.demo import load_tabular_demo
from lenskit.datasets import ML100K
from lenskit import crossfold as xf

import conf

ratings_mini = pd.read_csv(f"{conf.DATA_DIR}/mini_ml100k.csv", sep=',', encoding="latin-1")

ml100k = ML100K('ml-100k')
ratings_ml100k = ml100k.ratings
ratings_ml100k = ratings_ml100k[['user', 'item', 'rating']]

def create_sparse_matrix(ratings, file_path_train, file_path_test):     
    for train, test in xf.partition_users(ratings, 1, xf.SampleFrac(0.2)):
        user_item_matrix_train = train.pivot(*train.columns)
        user_item_matrix_train = user_item_matrix_train.fillna("")
        user_item_matrix_train.columns = user_item_matrix_train.columns.astype(str)
        user_item_matrix_train.to_csv(file_path_train)#, index=False)

        user_item_matrix_test = test.pivot(*test.columns)
        user_item_matrix_test = user_item_matrix_test.fillna("")
        user_item_matrix_test.columns = user_item_matrix_test.columns.astype(str)
        user_item_matrix_test.to_csv(file_path_test)#, index=False)

create_sparse_matrix(ratings_mini, f"{conf.DATA_DIR}/mini_ml100k_sparse_train.csv", f"{conf.DATA_DIR}/mini_ml100k_sparse_test.csv")
create_sparse_matrix(ratings_ml100k, f"{conf.DATA_DIR}/ml100k_sparse_train.csv",f"{conf.DATA_DIR}/ml100k_sparse_test.csv")