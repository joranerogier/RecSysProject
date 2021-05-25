from sdv.tabular import CTGAN
import numpy as np
import torch
import pandas as pd 
from lenskit.datasets import ML100K


# own scripts
import conf
from load_input_data import InputDataLoader

sample_subset_ratio = 1
model_file_name = "all_tau_0.165_500eps_500bs_str2"
ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
ctgan_model_path = f'{ctgan_dir}{model_file_name}.pkl'


# path to where to save new synthetic data
syn_sparse_path = f'{conf.SYN_DATA_DIR}all_tau_0.165_500eps_500bs_str2_.csv'

# Original training data (sparse)
data_loader = InputDataLoader("ml-100k", "")
input_data = data_loader.get_sparse_data()


# Set seed to ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load the saved model
m = CTGAN.load(ctgan_model_path)

# Create new data & save
sampling_set = int(len(input_data)*sample_subset_ratio)
new_data = m.sample(sampling_set)
print(new_data)

new_data.fillna(0).to_csv(f"{syn_sparse_path}", index=False)