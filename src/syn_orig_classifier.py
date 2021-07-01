import pandas as pd
from lenskit.datasets import ML100K
from sdv.metrics.tabular import LogisticDetection, SVCDetection

# import own scripts
import conf
from transform_data_representation import transform_dense_to_sparse_data, transform_sparse_to_dense_data


syn_path = f"{conf.SYN_DATA_DIR}750eps_non_partitioned.csv"

"""
Quick&Dirty solution for index problems that kept occuring for SVCDetection.
"""
syn_df = pd.read_csv(syn_path, sep=',', encoding="latin-1")
print(syn_df)
syn_df = transform_sparse_to_dense_data(syn_df)
print(syn_df)
syn_df = transform_dense_to_sparse_data(syn_df)
print(syn_df)

ml100k = ML100K('ml-100k')
ratings = ml100k.ratings
ratings = ratings[['user', 'item', 'rating']]

orig_df = transform_dense_to_sparse_data(ratings)

detector = SVCDetection.compute(orig_df, syn_df)
print(detector)