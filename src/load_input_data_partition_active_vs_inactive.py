import pandas as pd
from sdv.demo import load_tabular_demo
from timer import Timer
from lenskit.datasets import ML100K
from lenskit import crossfold as xf
import numpy as np

# import own scripts
import conf
from transform_data_representation import transform_to_sparse_data

class PartitionedInputDataLoader():
    def __init__(self, input_data="", input_path=""):
        self.max_active = 1000 # 30 for "own" dataset
        self.min_active = 350 # only 33 users # 5 for "own dataset"
        self.input_data = input_data
        
        if input_data == "demo":
            self.data = load_tabular_demo('student_placements') # demo data
        elif input_data == "ml-100k" or input_data=="own":
            #self.data_dir = f"{conf.DATA_DIR}/ml-100k/"
            if input_data=="own":
                self.input_path = input_path
            self.active_data_dense, self.inactive_data_dense, self.active_data_sparse, self.inactive_data_sparse = self.load_data()            
        else:
            print("Dataset was not provided or implementation is not yet prepared to handle this dataset.\nPlease try ml-100k or demo dataset.")


    def load_dense_data(self):
        """
        Load the input data (columns: user, item, rating)
        """
        if(self.input_data == "own"):
            ratings = pd.read_csv(self.input_path, sep=',', encoding="latin-1")
        else:
            ml100k = ML100K('ml-100k')
            ratings = ml100k.ratings
            ratings = ratings[['user', 'item', 'rating']]
        return ratings


    def partition_users(self, df):
        """
        Partition the input data into two subsets:
        - active users 
        - inactive users
        With this, we hope to enhance the user distribution in the sythetic data after training 
        two separate models on these subsets and combining them afterwards.
        """
        # Count the ratings per user, and get the corresponding set of active and inactive users
        df_users = df.groupby('user').size().reset_index(name='counts')

        df_active_counts = df_users[(df_users['counts'] >= self.min_active) & (df_users['counts'] <= 1000)]
        df_inactive_counts = df_users.loc[(df_users['counts'] < self.min_active)]
        active_users = np.array(df_active_counts['user'])
        inactive_users = np.array(df_inactive_counts['user'])

        df_active = df[df['user'].isin(active_users)]
        df_inactive = df[df['user'].isin(inactive_users)]

        print(f"Inactive users df: \n {df_inactive}")
        print(f"Active users df: \n {df_active}")
        return df_active, df_inactive

    def load_data(self):
        dense_data = self.load_dense_data()
        active_df_dense, inactive_df_dense = self.partition_users(dense_data)
        active_df_sparse = transform_to_sparse_data(active_df_dense)
        inactive_df_sparse = transform_to_sparse_data(inactive_df_dense)
        return active_df_dense, inactive_df_dense, active_df_sparse, inactive_df_sparse

    def get_sparse_data(self):
        return self.active_data_sparse, self.inactive_data_sparse

    def get_dense_data(self):
        return self.active_data_dense, self.inactive_data_dense
