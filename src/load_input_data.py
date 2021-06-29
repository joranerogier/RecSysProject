import pandas as pd
from sdv.demo import load_tabular_demo
from timer import Timer
from lenskit.datasets import ML100K

# import own scripts
import conf
from transform_data_representation import transform_dense_to_sparse_data, transform_dense_to_sparse_data_amazon

class InputDataLoader():
    def __init__(self, separation_char, input_data="", input_path=""):
        self.input_data = input_data
        if input_data == "demo":
            self.data = load_tabular_demo('student_placements') # demo data
        elif input_data == "ml-100k" or input_data=="own" or input_data=="amazon":
            #self.data_dir = f"{conf.DATA_DIR}/ml-100k/"
            if input_data=="own":
                self.input_path = input_path
            elif input_data=="amazon":
                self.input_data= f"{conf.DATA_DIR}/amazon/amazon_ratings.csv"
            self.sparse_data = self.load_sparse_data(separation_char)   
        else:
            print("Dataset was not provided or implementation is not yet prepared to handle this dataset.\nPlease try ml-100k or demo dataset.")

    def load_sparse_data(self, s):
        if(self.input_data == "own"):
            user_item_matrix = pd.read_csv(self.input_path, sep=s, encoding="latin-1")
            user_item_matrix = user_item_matrix[user_item_matrix.columns].astype(str)      
            print(user_item_matrix)
            #user_item_matrix = user_item_matrix.replace(0,"")
            #print(user_item_matrix)
            #user_item_matrix = transform_dense_to_sparse_data(ratings)
        elif (self.input_data == "ml-100k"):
            ml100k = ML100K('ml-100k')
            ratings = ml100k.ratings
            ratings = ratings[['user', 'item', 'rating']]
            user_item_matrix = transform_dense_to_sparse_data(ratings)
            #user_item_matrix[user_item_matrix.columns].astype(str)
            print(f"ui-matrix: {user_item_matrix}")
        else:
            df_dense = pd.read_csv(self.input_data, sep=s, encoding="latin-1")
            df_dense.columns = ['item', 'user', 'rating', 'timestamp']
            df_dense = df_dense[['item', 'user', 'rating']]
            user_item_matrix = transform_dense_to_sparse_data_amazon(df_dense)
            print(f"ui-matrix: {user_item_matrix}")

        return user_item_matrix    
   
    def get_sparse_data(self):
        return self.sparse_data

