from lenskit import crossfold as xf
from sdv.tabular import CTGAN
import pandas as pd
import csv
import conf

class CombinePartitionedSyntheticDataLoader():
    def __init__(self, c_date, syn_sparse_active, syn_sparse_inactive):
        self.current_date = c_date
        self.data_sparse_path_active = syn_sparse_active
        self.data_sparse_path_inactive = syn_sparse_inactive

        self.data_dense = self.transform_to_dense_combined_data()
        self.data_sparse_combined = self.create_sparse_combined()
        self.train_data_rec, self.test_data_rec = self.create_train_test_rec_data()

        self.current_date = c_date



    def transform_to_dense_combined_data(self):
        df_active = pd.read_csv(self.data_sparse_path_active, sep=',', encoding="latin-1").fillna("")
        df_inactive = pd.read_csv(self.data_sparse_path_inactive, sep=',', encoding="latin-1").fillna("")
        transformed_data = []

        nr_active_users = len(df_active)
        print(f"Nr of active users: {nr_active_users}")

        # Add the active users 
        for uid in range(len(df_active)):
            for item_id in df_active.columns:
                rating = df_active.iloc[uid][item_id]
                #print(f"user: {uid} - item: {item_id} - rating: {rating}")
                if rating != "":
                    user_id = int(uid)+1
                    sample = [user_id, int(item_id), int(rating)]
                    transformed_data.append(sample)

        # Add the inactive users
        # Add 'nr_active_users' + 1 to user id, to ensure that there are no duplicates.
        for uid in range(len(df_inactive)):
            for item_id in df_inactive.columns:
                rating = df_inactive.iloc[uid][item_id]
                #print(f"user: {uid} - item: {item_id} - rating: {rating}")
                if rating != "":
                    user_id = int(uid)+1+nr_active_users
                    sample = [user_id, int(item_id), int(rating)]
                    transformed_data.append(sample)
                
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
