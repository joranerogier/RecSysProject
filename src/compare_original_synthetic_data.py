"""
Script which (will) implement various functions to compare characteristics
of the original data with the newly created synthetic data.
"""
import pandas as pd
import csv
from check_existence_dir_csv import check_csv_path
import datetime

class DataComparison():
    def __init__(self, orig, syn, file_path):
        self.sparseness_orig = self.calculate_sparseness(orig)
        self.sparseness_syn = self.calculate_sparseness(syn)
        self.nr_users_orig = self.get_nr_users(orig)
        self.nr_users_syn = self.get_nr_users(syn)
        self.nr_items_orig = self.get_nr_items(orig)
        self.nr_items_syn = self.get_nr_items(syn)
        self.comparison_file_path = file_path
        check_csv_path(self.comparison_file_path, ['date', 'nr_users_orig', 'nr_user_syn', 'nr_items_orig', 'nr_items_syn', 'sparseness_orig', 'sparseness_syn'])

    def get_nr_non_empty_cells(self, df):
        # Returns the number of non-empty cells in the given dataset.
        df['nr_empty'] = df.iloc[:, 1:].eq("").sum(axis=1)
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
        sparsity = (1 + non_empty) / (self.get_nr_users(data) * self.get_nr_items(data))
        return sparsity

    def write_values_csv(self):
        current_date = datetime.date.today().strftime('%dm%y_%H%M')
        conn = open(self.comparison_file_path, 'a')
        writer = csv.writer(conn)
        writer.writerow([current_date, self.nr_users_orig, self.nr_users_syn, self.nr_items_orig, self.nr_items_syn, self.sparseness_orig, self.sparseness_syn])
        conn.close()

    def get_comparison_df(self):
        data = {'Characteristic': ['Nr_users', 'Nr_item', 'Sparseness'],
                'Orig': [self.nr_users_orig, self.nr_items_orig, self.sparseness_orig],
                'Syn': [self.nr_users_syn, self.nr_items_syn, self.sparseness_syn]}

        df = pd.DataFrame(data, columns = ['Characteristic', 'Orig', 'Syn'])
        return df

        
        

