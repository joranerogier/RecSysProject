import pandas as pd
from sdv.demo import load_tabular_demo
import time

# import own scripts
import conf

class InputDataLoader():
    def __init__(self, input_data=""):
        if input_data == "demo":
            self.data = load_tabular_demo('student_placements') # demo data
        elif input_data == "ml-100k":
            self.data_dir = f"{conf.DATA_DIR}/ml-100k/"
            self.data = self.load_ml100k_data()
        else:
            print("Dataset was not provided or implementation is not yet prepared to handle this dataset.\nPlease try ml-100k or demo dataset.")


    def load_ml100k_data(self):
        #FROM: http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/
        #u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        #users = pd.read_csv(f'{self.data_dir}u.user', sep='|', names=u_cols, encoding='latin-1')
        #m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        #movies = pd.read_csv(f'{self.data_dir}/u.item', sep='|', names=m_cols, usecols=range(5), encoding='latin-1')

        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        #ratings = pd.read_csv(f'{self.data_dir}u.data', sep='\t', names=r_cols, encoding='latin-1')
        
        # Own test input data to check if the pivot function works correctly.
        ratings = pd.read_csv(f"{conf.DATA_DIR}/test_input_data.csv", sep=';', names=r_cols, encoding="latin-1")
        ratings_no_timestamp = ratings[['user_id', 'movie_id', 'rating']]

        #movie_ratings = pd.merge(movies, ratings)
        #lens = pd.merge(movie_ratings, users)

        user_item_matrix = ratings_no_timestamp.pivot(*ratings_no_timestamp.columns)
        user_item_matrix = user_item_matrix.fillna("")
        user_item_matrix = user_item_matrix.applymap(str)
        print(user_item_matrix)
        
        return user_item_matrix


    def get_data(self):
        return self.data