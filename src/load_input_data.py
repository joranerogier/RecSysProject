import pandas as pd
from sdv.demo import load_tabular_demo
from timer import Timer

# import own scripts
import conf

class InputDataLoader():
    def __init__(self, input_data=""):
        if input_data == "demo":
            self.data = load_tabular_demo('student_placements') # demo data
        elif input_data == "ml-100k":
            self.data_dir = f"{conf.DATA_DIR}/ml-100k/"
            self.train_data_rec = self.load_train_rec_data()
            self.test_data_rec = self.load_test_rec_data()
            self.train_data_sparse = self.load_train_sparse_data()
            self.test_data_sparse = self.load_test_sparse_data()
             
        else:
            print("Dataset was not provided or implementation is not yet prepared to handle this dataset.\nPlease try ml-100k or demo dataset.")


    def load_train_sparse_data(self):     
        # Own test input data to check if the pivot function works correctly. 
        user_item_matrix = self.train_data_rec.pivot(*self.train_data_rec.columns)
        user_item_matrix = user_item_matrix.fillna("")
        user_item_matrix.columns = user_item_matrix.columns.astype(str)

        return user_item_matrix
    

    def load_test_sparse_data(self):      
        # Own test input data to check if the pivot function works correctly.
        user_item_matrix = self.test_data_rec.pivot(*self.test_data_rec.columns)
        user_item_matrix = user_item_matrix.fillna("")
        user_item_matrix.columns = user_item_matrix.columns.astype(str)
             
        return user_item_matrix


    def load_train_rec_data(self):
        #FROM: http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

        #ratings = pd.read_csv(f'{self.data_dir}u1.base', sep='\t', names=r_cols, encoding='latin-1')
        ratings = pd.read_csv(f"{conf.DATA_DIR}/test_train_input_data.csv", sep=';', names=r_cols, encoding="latin-1")
        
        ratings_no_timestamp = ratings[['user_id', 'movie_id', 'rating']]
        ratings_no_timestamp = ratings_no_timestamp.rename(columns={'user_id': 'user', 'movie_id': 'item'})

        return ratings_no_timestamp

    def load_test_rec_data(self):
        #FROM: http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        
        #ratings = pd.read_csv(f'{self.data_dir}u1.test', sep='\t', names=r_cols, encoding='latin-1')
        ratings = pd.read_csv(f"{conf.DATA_DIR}/test_test_input_data.csv", sep=';', names=r_cols, encoding="latin-1")
       
        ratings_no_timestamp = ratings[['user_id', 'movie_id', 'rating']]
        ratings_no_timestamp = ratings_no_timestamp.rename(columns={'user_id': 'user', 'movie_id': 'item'})

        return ratings_no_timestamp

    def get_train_rec_data(self):
        return self.train_data_rec
    
    def get_test_rec_data(self):
        return self.train_data_rec

    def get_train_sparse_data(self):
        return self.train_data_sparse

    def get_test_sparse_data(self):
        return self.test_data_sparse