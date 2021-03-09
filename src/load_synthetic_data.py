from lenskit import crossfold as xf
from sdv.tabular import CTGAN

class SyntheticDataLoader():
    def __init__(self, ctgan_model, nr_samples):
        self.data_rec = self.load_synthetic_data(ctgan_model, nr_samples)
        self.train_data_rec, self.test_data_rec = self.create_train_test_rec_data()

    def load_synthetic_data(self, m, s):
        loaded = CTGAN.load(m)
        new_data = loaded.sample(s)
        return new_data

    def create_train_test_rec_data(self):
        # For now, no cross-validation, just split the data into 1 train and 1 test set.
        for train, test in xf.partition_users(self.data_rec, 1, xf.SampleFrac(0.2)):
            print("created train-test split")
            print(train.head())
            print(test.head())
            return train, test
 
    def get_train_test_data(self):
        return self.train_data_rec, self.test_data_rec
