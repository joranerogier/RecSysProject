from lenskit.datasets import ML100K
import pandas as pd
import csv 
import conf

nr_users_subset = 5

ml100k = ML100K('ml-100k')
ratings = ml100k.ratings
ratings = ratings[['user', 'item', 'rating']]
subset_ratings = ratings[ratings['user'] <= nr_users_subset]

file_path = f"{conf.DATA_DIR}/mini_ml100k_{nr_users_subset}_users.csv"

subset_ratings.to_csv(file_path, index=False)
