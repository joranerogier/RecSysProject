from lenskit.datasets import ML100K
import pandas as pd
import csv 
import conf

nr_users = 10
nr_items = 25

ml100k = ML100K('ml-100k')
ratings = ml100k.ratings
ratings = ratings[['user', 'item', 'rating']]
subset_ratings = ratings[ratings['user'] <= nr_users]
subset_ratings_items = subset_ratings[subset_ratings['item'] <= nr_items]

file_path = f"{conf.DATA_DIR}/mini_ml100k_user_{nr_users}_item_{nr_items}.csv"

subset_ratings_items.to_csv(file_path, index=False)
