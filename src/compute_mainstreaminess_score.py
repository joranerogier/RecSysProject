"""
This script computes the mainstreaminess of users.
INPUT: User-Item-Rating matrix
OUTPUT: Vector with the M^global_R,APC mainstreaminess values, adopted to movie recommendations [see "Global and country-specific mainstreaminess measures" article] TODO: site correctly

Instead of Artist Play Counts, we will use the rating information "Item Rating" (IT)
We have to keep in mind that, for example, if an item has been watched 5 times with a rating of 1, this will result in the same value as for an item rated once with a 5..
However, as a first implementation, I will try to attack the problem in this way, hoping that "high mainstream" subset can be found in this way.
- Probably, movies that have relatively many low values will also be watched less
"""
from lenskit.datasets import ML100K
from transform_data_representation import transform_dense_to_sparse_data
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Load input data
ml100k = ML100K('ml-100k')
ratings = ml100k.ratings
ratings = ratings[['user', 'item', 'rating']]
user_item_matrix = ratings.pivot(*ratings.columns)
user_item_matrix = user_item_matrix.fillna(0)
user_item_matrix.columns = user_item_matrix.columns.astype(str)
df = pd.DataFrame(user_item_matrix)

# Create Global Item Ranking (GIR), looping over all items, count how often the item is watched
GIR = {}
for col in df:
    # does not take into account movies that have not been watched
    summed_rating = df[col].sum()

    # only take into account items that have been rated more than 5 times
    if (df[col].astype(bool).sum(axis=0) > 100):
        GIR[col] = summed_rating

ranks_GIR = {k: v for k,v in sorted(GIR.items(), key=lambda item: item[1], reverse=True)}
ranks_GIR_items = [*ranks_GIR]
print(len(ranks_GIR_items))

# Compute user mainstreaminess (UM)
user_mainstreamnesses = {}
taus = []
for uid in range(len(df)):
    user_dict = {}
    for item_id in df.columns:
        rating = df.iloc[uid][item_id]

        # only take into account items that have been rated > 5 times
        if item_id in GIR:
            user_dict[item_id] = rating
    ranks_user = {k: v for k,v in sorted(user_dict.items(), key=lambda item: item[1], reverse=True)}
    ranks_user_items = [*ranks_user]
    # compute the mainstreamness measure with Kendall's rank-order correlation
    tau, p_value = stats.kendalltau(ranks_GIR_items, ranks_user_items)
    user_mainstreamnesses[uid] = [tau, p_value]
    taus.append(round(tau, 3))

#print(user_mainstreamnesses)
#print(taus)

#  Plot taus to investigate the values
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

plt.hist(taus)
plt.xlabel("Kendall's tau")
plt.ylabel("Counts")
plt.savefig(f'{ROOT_DIR}/output/imgs/taus_histogram_100ratings.png')
