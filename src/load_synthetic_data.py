from lenskit import crossfold as xf
from sdv.tabular import CTGAN
import pandas as pd
import csv

#import own scripts
import conf
from transform_data_representation import transform_sparse_to_dense_data, dense_to_csv

class SyntheticDataLoader():
    """
    This class takes the synthetic sparse data created on a specific date, 
    transforms this into a dense representation (which is saved), and splits the data into train/test set.
    """
    def __init__(self, c_date, bs, epochs, sparse_filename):
        if c_date == "":
            self.data_sparse_path = sparse_filename
        else:
            self.data_sparse_path = f'{conf.SYN_DATA_DIR}syn_sparse_{c_date}.csv'
        self.sparse_df = pd.read_csv(self.data_sparse_path, sep=',', encoding="latin-1").fillna("")
        
        # Transform the data to a dense representation
        self.data_dense = transform_sparse_to_dense_data()
        # Save dense data to csv
        dense_to_csv(self.data_dense, c_date, epochs, bs)

        # Split the dataset
        self.train_data_rec, self.test_data_rec = self.create_save_train_test_rec_data(c_date, epochs, bs)

    def create_save_train_test_rec_data(self, current_date, epochs, bs):
        # For now, no cross-validation, just split the data into 1 train and 1 test set.
        for i, tp in enumerate(xf.partition_users(data=self.data_dense, partitions=1, method=xf.SampleN(5), rng_spec=1)):
            train = tp.train
            test = tp.test
            train.to_csv(f'{conf.SYN_DATA_DIR}syn_train_{current_date}_{epochs}eps_{bs}bs.csv')
            test.to_csv(f'{conf.SYN_DATA_DIR}syn_test_{current_date}_{epochs}eps_{bs}bs.csv')
            return train, test
 
    def get_train_test_data(self):
        # Return the train/test split
        return self.train_data_rec, self.test_data_rec
