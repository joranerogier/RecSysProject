from lenskit import crossfold as xf
from sdv.tabular import CTGAN
import pandas as pd
import csv
import conf

class SyntheticDataLoader():
    def __init__(self, c_date):
        self.current_date = c_date
        self.data_sparse_path = f'{conf.SYN_DATA_DIR}{c_date}.csv'
        self.data_dense = self.transform_to_dense_data()
        self.train_data_rec, self.test_data_rec = self.create_train_test_rec_data()

        self.current_date = c_date



    def transform_to_dense_data(self):
        d = pd.read_csv(self.data_sparse_path, sep=',', encoding="latin-1").fillna("")
        print(d.fillna(""))
        transformed_data = []
        for uid in range(len(d)):
            for item_id in d.columns:
                rating = d.iloc[uid][item_id]
                #print(f"user: {uid} - item: {item_id} - rating: {rating}")
                if rating != "":
                    user_id = int(uid)+1
                    sample = [user_id, int(item_id), int(rating)]
                    transformed_data.append(sample)
                
        df = pd.DataFrame(transformed_data, columns =['user', 'item', 'rating'])
        df.to_csv(f'{conf.SYN_DATA_DIR}syn_complete_{self.current_date}.csv')
        # TODO: Save to csv
        #print(df)
        return df

    def create_train_test_rec_data(self):
        # For now, no cross-validation, just split the data into 1 train and 1 test set.
       
        '''# Manel:
        for i, tp in enumerate(xf.partition_users(ratings, 1, xf.SampleN(5))):
            tp.train.to_csv('ml1m_original_half_train-%d.csv' % (i,), index= False)
            tp.test.to_csv('ml1m_original_half_test-%d.csv' % (i,), index= False)'''
        for i, tp in enumerate(xf.partition_users(data=self.data_dense, partitions=1, method=xf.SampleN(5), rng_spec=1)):
            train = tp.train
            test = tp.test
            train.to_csv(f'{conf.SYN_DATA_DIR}syn_train_{self.current_date}.csv')
            test.to_csv(f'{conf.SYN_DATA_DIR}syn_test_{self.current_date}.csv')

            return train, test
 
    def get_train_test_data(self):
        return self.train_data_rec, self.test_data_rec
