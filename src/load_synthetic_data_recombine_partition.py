from lenskit import crossfold as xf
from sdv.tabular import CTGAN
import pandas as pd
import csv

# Import own scripts
import conf
from sparse_to_dense import transform_sparse_to_dense_data
from dense_to_sparse import transform_dense_to_sparse_data

class CombinePartitionedSyntheticDataLoader():
    def __init__(self, c_date, syn_sparse_active, syn_sparse_inactive):
        self.current_date = c_date
        self.data_sparse_path_active = syn_sparse_active
        self.data_sparse_path_inactive = syn_sparse_inactive

        self.data_dense = self.transform_to_dense_combined_data()
        self.data_sparse_combined = transform_dense_to_sparse_data(self.data_dense)
        self.train_data_rec, self.test_data_rec = self.create_train_test_rec_data()


    def transform_to_dense_combined_data(self):
        df_active = pd.read_csv(self.data_sparse_path_active, sep=',', encoding="latin-1").fillna("")
        df_inactive = pd.read_csv(self.data_sparse_path_inactive, sep=',', encoding="latin-1").fillna("")
        transformed_data = []

        # Add the active users 
        active_dense = transform_sparse_to_dense_data(df_active)
        transformed_data.append(active_dense)

        # Add the inactive users
        inactive_dense = transform_sparse_to_dense_data(df_inactive)
        transformed_data.append(inactive_dense)
        
        df = pd.DataFrame(transformed_data, columns =['user', 'item', 'rating'])
        df.to_csv(f'{conf.SYN_DATA_DIR}syn_complete_{self.current_date}_combined_partition.csv')
        print(f"Combined df: \n {df}")
        return df

    def create_train_test_rec_data(self):
        # For now, no cross-validation, just split the data into 1 train and 1 test set.
        for i, tp in enumerate(xf.partition_users(data=self.data_dense, partitions=1, method=xf.SampleN(5), rng_spec=1)):
            train = tp.train
            test = tp.test
            train.to_csv(f'{conf.SYN_DATA_DIR}syn_train_{self.current_date}.csv')
            test.to_csv(f'{conf.SYN_DATA_DIR}syn_test_{self.current_date}.csv')
        return train, test

    def create_sparse_combined(self):
        user_item_matrix = self.data_dense.pivot(*self.data_dense.columns)
        user_item_matrix = user_item_matrix.fillna("")
        user_item_matrix.columns = user_item_matrix.columns.astype(str)
        print(user_item_matrix.head())
        return user_item_matrix
 
    def get_train_test_data(self):
        return self.train_data_rec, self.test_data_rec

    def get_sparse_combined(self):
        return self.data_sparse_combined
    
    def get_dense_combined(self):
        return self.data_dense
