import pandas as pd
from sdv.demo import load_tabular_demo
from timer import Timer
from lenskit.datasets import ML100K
from lenskit import crossfold as xf

# import own scripts
import conf
from dense_to_sparse import transform_dense_to_sparse_data

class InputDataLoader():
    def __init__(self, input_data="", input_path=""):
        self.input_data = input_data
        if input_data == "demo":
            self.data = load_tabular_demo('student_placements') # demo data
        elif input_data == "ml-100k" or input_data=="own":
            #self.data_dir = f"{conf.DATA_DIR}/ml-100k/"
            if input_data=="own":
                self.input_path = input_path
            self.sparse_data = self.load_sparse_data()            
        else:
            print("Dataset was not provided or implementation is not yet prepared to handle this dataset.\nPlease try ml-100k or demo dataset.")

    def load_sparse_data(self):
        if(self.input_data == "own"):
            ratings = pd.read_csv(self.input_path, sep=',', encoding="latin-1")
        else:
            ml100k = ML100K('ml-100k')
            ratings = ml100k.ratings
            ratings = ratings[['user', 'item', 'rating']]

        user_item_matrix = transform_dense_to_sparse_data(ratings)
        return user_item_matrix    
   
    def get_sparse_data(self):
        return self.sparse_data
