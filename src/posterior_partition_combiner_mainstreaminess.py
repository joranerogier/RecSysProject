from sdv.tabular import CTGAN
import numpy as np
import torch
import pandas as pd 
import random

# own scripts
import conf
from load_synthetic_data_recombine_partition import CombinePartitionedSyntheticDataLoader

prefixes = ['all', 'beyond', 'combined']

model_file_name = "_tau_0.07_l20_1000eps_300bs"
datetime_now = '140621_1007'
ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
all_ctgan_model_path = f'{ctgan_dir}{prefixes[0]}{model_file_name}.pkl'
beyond_mainstream_ctgan_model_path = f'{ctgan_dir}{prefixes[1]}{model_file_name}.pkl'

# path to where to save new synthetic data
syn_sparse_path_all = f'{conf.SYN_DATA_DIR}syn_sparse_{prefixes[0]}{model_file_name}.csv'
syn_sparse_path_beyond_mainstream = f'{conf.SYN_DATA_DIR}syn_sparse_{prefixes[1]}{model_file_name}.csv'
syn_sparse_path_combined = f'{conf.SYN_DATA_DIR}syn_sparse_{prefixes[2]}{model_file_name}.csv'
syn_dense_path_combined = f'{conf.SYN_DATA_DIR}syn_dense_{prefixes[2]}{model_file_name}.csv'

# Original training data (partitioned)
orig_sparse_all = f"{conf.PARTITIONED_MAINSTREAM_DATA_DIR}orig_sparse_all_tau_tau_0.07_l20.csv"
orig_sparse_beyond_mainstream = f"{conf.PARTITIONED_MAINSTREAM_DATA_DIR}orig_sparse_beyond_tau_tau_0.07_l20.csv"
df_orig_all = pd.read_csv(orig_sparse_all, sep=',', encoding="latin-1").fillna(0)
df_orig_beyond = pd.read_csv(orig_sparse_beyond_mainstream, sep=',', encoding="latin-1").fillna(0)

# Set seed to ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load the saved models
all_model = CTGAN.load(all_ctgan_model_path)
beyond_mainstream_model = CTGAN.load(beyond_mainstream_ctgan_model_path)

# Create new data & save
new_data_all = all_model.sample(len(df_orig_all))
new_data_beyond = beyond_mainstream_model.sample(len(df_orig_beyond))
#print(f"NEW ALL DATA HERE: \n{new_data_all}")

# Drop len(orig_sparse_beyond_mainstream) from the "all" data to end with the same amount of users as the original data
#print(len(new_data_beyond))
random_indexes = random.sample(range(len(df_orig_all)), len(df_orig_beyond))
#print(random_indexes)
new_data_all.drop(random_indexes, 0, inplace=True)

#print(f"SHORTENED ALL DATA HERE:\n{new_data_all}")

## Change user-ids in beyond-data to start after the 'all-data', removing the overlap

new_data_all.fillna(0).to_csv(f"{syn_sparse_path_all}", index=False)
new_data_beyond.fillna(0).to_csv(f"{syn_sparse_path_beyond_mainstream}", index=False)
#print(f"NEW DATA BEYOND: \n{new_data_beyond}")

# Combine active/inactive partition
combine_syn_data_loader = CombinePartitionedSyntheticDataLoader(datetime_now, syn_sparse_path_beyond_mainstream, syn_sparse_path_all)
combined_dense = combine_syn_data_loader.get_dense_combined()
combined_sparse = combine_syn_data_loader.get_sparse_combined()
print(f"Combined data: \n{combined_sparse}")

combined_sparse.fillna(0).to_csv(f"{syn_sparse_path_combined}", index=False)
combined_dense.fillna(0).to_csv(f"{syn_dense_path_combined}", index=False)