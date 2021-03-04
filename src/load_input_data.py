import pandas as pd
from sdv.demo import load_tabular_demo
from timer import Timer
from lenskit.datasets import ML100K
from lenskit import crossfold as xf

# import own scripts
import conf

class InputDataLoader():
    def __init__(self, input_data=""):
        self.input_data = input_data
        if input_data == "demo":
            self.data = load_tabular_demo('student_placements') # demo data
        elif input_data == "ml-100k" or input_data=="own":
            #self.data_dir = f"{conf.DATA_DIR}/ml-100k/"
            self.train_data_rec, self.test_data_rec = self.load_train_test_rec_data()
            #self.test_data_rec = self.load_test_rec_data()
            self.train_data_sparse = self.load_train_sparse_data()
            self.test_data_sparse = self.load_test_sparse_data()
             
        else:
            print("Dataset was not provided or implementation is not yet prepared to handle this dataset.\nPlease try ml-100k or demo dataset.")


    def load_train_sparse_data(self):     
        # Own test input data to check if the pivot function works correctly. 
        '''user_item_matrix = self.train_data_rec.pivot(*self.train_data_rec.columns)
        user_item_matrix = user_item_matrix.fillna("")
        user_item_matrix.columns = user_item_matrix.columns.astype(str)
        print(user_item_matrix.head())'''
        
        # The sparse matrices for the ml100k and mini subset were stored locally, 
        # since computing these took a lot of time. 
        if self.input_data == "ml-100k":
            user_item_matrix = f"{conf.DATA_DIR}/ml100k_sparse_train.csv"
        elif self.input_data == "own":
            user_item_matrix = f"{conf.DATA_DIR}/mini_ml100k_sparse_train.csv"
        return user_item_matrix
    

    def load_test_sparse_data(self):      
        # Own test input data to check if the pivot function works correctly.
        '''user_item_matrix = self.test_data_rec.pivot(*self.test_data_rec.columns)
        user_item_matrix = user_item_matrix.fillna("")
        user_item_matrix.columns = user_item_matrix.columns.astype(str)  
        print(user_item_matrix.head()) '''
        if self.input_data == "ml-100k":
            user_item_matrix = f"{conf.DATA_DIR}/ml100k_sparse_test.csv"
        elif self.input_data == "own":
            user_item_matrix = f"{conf.DATA_DIR}/mini_ml100k_sparse_test.csv"
        return user_item_matrix


    def load_train_test_rec_data(self):
        '''
        if (self.input_data == "own"):
            r_cols = ['user', 'item', 'rating', 'unix_timestamp']
            ratings_train = pd.read_csv(f"{conf.DATA_DIR}/test_train_input_data.csv", sep=';', names=r_cols, encoding="latin-1")
            ratings_test = pd.read_csv(f"{conf.DATA_DIR}/test_test_input_data.csv", sep=';', names=r_cols, encoding="latin-1")
            return ratings_train[['user', 'item', 'rating']], ratings_test[['user', 'item', 'rating']]
        
        else:
        '''

        if(self.input_data == "own"):
            ratings = pd.read_csv(f"{conf.DATA_DIR}/mini_ml100k.csv", sep=',', encoding="latin-1")
        else:
            #ratings = pd.read_csv(f'{self.data_dir}u1.base', sep='\t', names=r_cols, encoding='latin-1')
            ml100k = ML100K('ml-100k')
            ratings = ml100k.ratings
            ratings = ratings[['user', 'item', 'rating']]
            
        for train, test in xf.partition_users(ratings, 1, xf.SampleFrac(0.2)):
            print("created train-test split")
            print(train.head())
            print(test.head())
            return train, test
        
    def get_train_rec_data(self):
        return self.train_data_rec
    
    def get_test_rec_data(self):
        return self.train_data_rec

    def get_train_sparse_data(self):
        return self.train_data_sparse

    def get_test_sparse_data(self):
        return self.test_data_sparse