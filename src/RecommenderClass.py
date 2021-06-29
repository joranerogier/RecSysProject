"""
This is the recommender class, which can use any of the following three algorithms:
* itemKNN
* userKNN
* BPRMF
""" 

import warnings
from sdv.tabular import CTGAN
from lenskit.algorithms import Recommender, als, item_knn, user_knn, tf
from lenskit import algorithms
from lenskit.metrics.predict import rmse
from load_input_data import InputDataLoader
from lenskit import batch, topn, util
#import tensorflow as tf

# import own scripts
import conf
#from check_existence_dir_csv import check_dir, check_csv_path

class RecommenderSystem():
    def __init__(self, train_data, test_data):
        self.num_recs = 10
        self.max_nbrs = 15 # "reasonable default"
        self.min_nbrs = 3 # "reasonable default"
        """
        For the moment, no cross validation is used.
        Thus, the train & test data are just set globally.
        """
        self.train, self.test = train_data, test_data


    def eval(self, aname, algo):
        fittable = util.clone(algo)
        fittable = Recommender.adapt(fittable)
        #print(self.train)
        fittable.fit(self.train)
        users = self.test.user.unique()
        recs = batch.recommend(fittable, users, self.num_recs)
        recs['Algorithm'] = aname
        return recs

    def analyze_performance(self, recs):
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg)
        results = rla.compute(recs, self.test)
        #print(results.head)
        print(results.groupby('Algorithm').ndcg.mean())
        return results

    def itemKNN(self, nnbrs, aggregate, center, min_nbrs=3):
        algoname = "itemKNN"
        item_item = item_knn.ItemItem(nnbrs=nnbrs, 
                                        min_nbrs=min_nbrs, 
                                        aggregate=aggregate, 
                                        center=center)
        eval = self.eval(algoname, item_item)
        print("ItemKNN was fitted.")
        return eval

    def userKNN(self, nnbrs, aggregate, center, min_nbrs=3):
        algoname = "userKNN"
        user_user = user_knn.UserUser(nnbrs=nnbrs, 
                                        min_nbrs=min_nbrs, 
                                        aggregate=aggregate, 
                                        center=center)
        eval = self.eval(algoname, user_user)
        print("UserKNN was fitted.")
        return eval

    def BPRMF(self, nr_features=50, eps=1, bs=500):
        # Bayesian personalized ranking matrix factorization
        algoname = "BPRMF"
        bprmf = tf.BPR(features=nr_features, epochs=eps, batch_size=bs)# sensible default value
        eval = self.eval(algoname, bprmf)
        print("BPRMF was fitted.")
        return eval
