"""
Script which (will) implement various functions to compare characteristics
of the original data with the newly created synthetic data.
"""
import pandas as pd
import csv
from check_existence_dir_csv import check_csv_path
import datetime

import conf
from lenskit.datasets import ML100K
from transform_data_representation import transform_dense_to_sparse_data

class DataComparison():
    def __init__(self, orig, syn):
        self.sparseness_orig = self.calculate_sparseness(orig)
        self.sparseness_syn = self.calculate_sparseness(syn)
        self.nr_users_orig = self.get_nr_users(orig)
        self.nr_users_syn = self.get_nr_users(syn)
        self.nr_items_orig = self.get_nr_items(orig)
        self.nr_items_syn = self.get_nr_items(syn)

    def get_nr_non_empty_cells(self, df):
        # Returns the number of non-empty cells in the given dataset.
        df['nr_empty'] = df.iloc[:, 1:].eq(0).sum(axis=1)
        nr_non_empty = df['nr_empty'].sum()
        print(f"Nr of non-empty cells: {nr_non_empty}")
        return nr_non_empty

    def get_nr_users(self, df):
        return len(df)

    def get_nr_items(self, df):
        return len(df.columns)

    def calculate_sparseness(self, data):
        #Sparsity = 1 + len (data) / (#users * #items)
        non_empty = self.get_nr_non_empty_cells(data)
        sparsity = 1 - (non_empty / (self.get_nr_users(data) * self.get_nr_items(data)))
        return sparsity

    def get_values_csv(self):
        values = [self.nr_users_orig, self.nr_users_syn, self.nr_items_orig, self.nr_items_syn, self.sparseness_orig, self.sparseness_syn]
        return values

    def get_comparison_df(self):
        data = {'Characteristic': ['Nr_users', 'Nr_items', 'Sparseness'],
                'Orig': [self.nr_users_orig, self.nr_items_orig, self.sparseness_orig],
                'Syn': [self.nr_users_syn, self.nr_items_syn, self.sparseness_syn]}

        df = pd.DataFrame(data, columns = ['Characteristic', 'Orig', 'Syn'])
        return df


syn_dense = "C:/Users/Jorane Rogier/Documents/studie/year2/Research Internship/RecSysProject/data/lastfm/syn_dense_combined_tau_0.18_impl60_750eps_300bs.csv"
syn_dense = pd.read_csv(syn_dense, sep=',', encoding="latin-1").fillna(0)
s = transform_dense_to_sparse_data(syn_dense)
print(s)

# Load original implicit input data
orig_lastfm_path = "C:/Users/Jorane Rogier/Documents/studie/year2/Research Internship/RecSysProject/output/partitioned_mainstreaminess_data/orig_dense_combined_tau_0.18_playcounts_impl60_lastfm_items_removed.csv"
orig_data = pd.read_csv(orig_lastfm_path, sep=',')
orig_data = orig_data[['user', 'item', 'rating']]
orig_data.loc[orig_data.rating > 0, "rating"] = 1
print(orig_data)
o = transform_dense_to_sparse_data(orig_data)
print(o)

data_comp = DataComparison(o, s)
comp = data_comp.get_comparison_df()
print(comp)