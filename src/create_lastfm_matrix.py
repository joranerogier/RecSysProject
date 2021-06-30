"""
This script reads the original LastFM dataset csv file, 
transforms it to a user-item matrix, removes the users that have less then 20 ratings, 
and saves the resulting user-item matrix locally. 
"""

from lenskit import crossfold as xf
from sdv.tabular import CTGAN
import pandas as pd
import csv

#import own scripts
import conf
from transform_data_representation import transform_dense_to_sparse_data

#https://curiousily.com/posts/music-artist-recommender-system-using-stochastic-gradient-descent/

# Load input data
plays = pd.read_csv(f'{conf.DATA_DIR}lastfm/user_artists.dat', sep='\t')
artists = pd.read_csv(f'{conf.DATA_DIR}lastfm/artists.dat', sep='\t', usecols=['id', 'name'])

df = pd.merge(artists, plays, how='inner', left_on='id', right_on='artistID')

# Rename columns to match the standard used in movielens data
df = df.rename(columns={'weight': 'rating', 'userID': 'user', 'artistID': 'item'})

# Only keep artistID, not name
df = df[['user', 'item', 'rating']]
user_item_matrix = transform_dense_to_sparse_data(df)

user_activities_count = []
for index, row in user_item_matrix.iterrows():
    nr_activities = len(row.to_numpy().nonzero()[0])
    user_activities_count.append(nr_activities)

# Remove users with less than 20 ratings
print(user_item_matrix)
user_item_matrix['non-zero'] = user_activities_count
user_item_matrix = user_item_matrix[user_item_matrix['non-zero'] > 20]
print(user_item_matrix)

user_item_matrix.to_csv(f'{conf.DATA_DIR}lastfm/lastfm_sparse.csv')