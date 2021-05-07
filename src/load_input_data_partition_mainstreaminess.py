import pandas as pd
import numpy as np

# import own scripts
import conf
from transform_data_representation import transform_dense_to_sparse_data

class MainstreaminessPartitionedInputDataLoader():
    def __init__(self, input_path_bm="", input_path_m="", output_path_bm="", output_path_m=""):
        self.ratings_bm, self.ratings_all = self.load_data(input_path_bm, input_path_m)
        self.save_data(output_path_bm, output_path_m)      

    def load_data(self, input_path_bm, input_path_m):
        """
        Load the input data (columns: user, item, rating)
        """
        #load beyond mainstream data, transform to sparse data
        ratings_bm = pd.read_csv(input_path_bm, sep=',', encoding="latin-1", dtype={'user': np.int32, 'item': np.int32, 'rating': np.int32})
        ratings_bm_sparse = transform_dense_to_sparse_data(ratings_bm)
        ratings_bm_sparse = ratings_bm_sparse[ratings_bm_sparse.columns].astype(int)
        #load mainstream data, transform to sparse data
        ratings_m = pd.read_csv(input_path_m, sep=',', encoding="latin-1", dtype={'user': np.int32, 'item': np.int32, 'rating': np.int32})
        ratings_m_sparse = transform_dense_to_sparse_data(ratings_m)
        ratings_m_sparse = ratings_m_sparse[ratings_m_sparse.columns].astype(int)

        # combine mainstream and beyond mainstream to enable mainstream users to learn from beyond-mainstream users during training
        ratings_all = pd.concat([ratings_bm_sparse, ratings_m_sparse], axis=0)
        #shuffle the ratings
        ratings_all = ratings_all.sample(frac=1)
        print(ratings_bm_sparse)
        return ratings_bm_sparse, ratings_all
    
    def save_data(self, output_bm, output_m):
        """
        Save the created dataframes to csv files
        """
        self.ratings_bm.to_csv(output_bm)
        self.ratings_all.to_csv(output_all)


input_bm = f'{conf.OUTPUT_DIR}partitioned_mainstreaminess_data/orig_dense_beyond_mainstream_tau_0.165.csv'
input_m = f'{conf.OUTPUT_DIR}partitioned_mainstreaminess_data/orig_dense_mainstream_tau_0.165.csv'

output_bm = f'{conf.OUTPUT_DIR}partitioned_mainstreaminess_data/orig_sparse_beyond_mainstream_tau_0.165.csv'
output_all = f'{conf.OUTPUT_DIR}partitioned_mainstreaminess_data/orig_sparse_all_tau_0.165.csv'

mp = MainstreaminessPartitionedInputDataLoader(input_bm, input_m, output_bm, output_all)

