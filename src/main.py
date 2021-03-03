# TODO: add arguments to change variables directly.
import pandas as pd
import conf 

# Import own packages
from CTGAN_trainer import TrainModel
from load_input_data import InputDataLoader


ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"

data_loader = InputDataLoader("ml-100k")
train_sparse = data_loader.get_train_sparse_data()
test_sparse = data_loader.get_test_sparse_data()
train_rec = data_loader.get_train_rec_data()
test_rec = data_loader.get_test_rec_data()

#m = TrainModel(dn="test eval", ctgan_dir=ctgan_dir)
#m.build_model(data)

