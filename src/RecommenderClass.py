"""
This is the recommender class, which can use any of the following three algorithms:
* itemKNN
* userKNN
* BPRMF
""" 

import warnings
from sdv.tabular import CTGAN
from lenskit.algorithms import Recommender, als, item_knn, user_knn
from lenskit.metrics.predict import rmse
from load_input_data import InputDataLoader
from lenskit import batch, topn, util

# import own scripts
import conf
from check_existence_dir_csv import check_dir, check_csv_path

class RecommenderSystem():
    def __init__(self):
        self.num_recs = 10
        self.max_nbrs = 15 # "reasonable default"
        self.min_nbrs = 3 # "reasonable default"

    def itemKNN(self):
        item_knn.ItemItem()
        pass

    def fitUserKNN(self, data):
        user_user = user_knn.UserUser(self.max_nbrs, self.min_nbrs)
        algo = Recommender.adapt(user_user)
        algo.fit(data)
        print("UserKNN was fitted.")
    
    def predictUserKNN(self, algo):
        recs = algo.recommend(-1, self.num_recs, )


    def BPRMF(self):
        # Bayesian personalized ranking matrix factorization
        pass

num_recs = 5
max_nbrs = 15 # "reasonable default"
min_nbrs = 3 # "reasonable default"

data_loader = InputDataLoader("ml-100k")
train_rec = data_loader.get_train_rec_data()
test_rec = data_loader.get_test_rec_data()

user_user = user_knn.UserUser(max_nbrs, min_nbrs)
fittable = util.clone(user_user)
fittable = Recommender.adapt(fittable)
fittable.fit(train_rec)
users = test_rec.user.unique()
print(users)
recs = batch.recommend(fittable, users, num_recs)
print(recs)
