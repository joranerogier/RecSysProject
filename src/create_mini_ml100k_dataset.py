from lenskit.datasets import ML100K
import pandas as pd
import csv 
import conf

ml100k = ML100K('ml-100k')
ratings = ml100k.ratings
ratings = ratings[['user', 'item', 'rating']]
subset_ratings = ratings[ratings['user'] <= 25]

file_path = f"{conf.DATA_DIR}/mini_ml100k.csv"

subset_ratings.to_csv(file_path, index=False)
