from sdv.tabular import CTGAN
import numpy as np
import torch
import pandas as pd 

# own scripts
import conf
from  load_input_data_partition_active_vs_inactive import PartitionedInputDataLoader
from load_synthetic_data_recombine_partition import CombinePartitionedSyntheticDataLoader

model_file_name = "500eps_bs300_partitioned"
datetime_now = '070421_0935'
ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
active_ctgan_model_path = f'{ctgan_dir}{model_file_name}_active.pkl'
inactive_ctgan_model_path = f'{ctgan_dir}{model_file_name}_inactive.pkl'

# path to where to save new synthetic data
syn_sparse_path_active = f'{conf.SYN_DATA_DIR}syn_sparse_{datetime_now}_active.csv'
syn_sparse_path_inactive = f'{conf.SYN_DATA_DIR}syn_sparse_{datetime_now}_inactive.csv'

# Original training data (partitioned)
orig_sparse_active = f"{conf.PARTITIONED_DATA_DIR}orig_sparse_{datetime_now}_active.csv"
orig_sparse_inactive = f"{conf.PARTITIONED_DATA_DIR}orig_sparse_{datetime_now}_inactive.csv"
df_orig_active = pd.read_csv(orig_sparse_active, sep=',', encoding="latin-1").fillna("")
df_orig_inactive = pd.read_csv(orig_sparse_inactive, sep=',', encoding="latin-1").fillna("")

# Set seed to ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load the saved models
active_model = CTGAN.load(active_ctgan_model_path)
inactive_model = CTGAN.load(inactive_ctgan_model_path)

# Create new data & save
new_data_active = active_model.sample(len(df_orig_active))
new_data_inactive = inactive_model.sample(len(df_orig_inactive))
print(f"NEW ACTIVE DATA HERE: {new_data_active}")
new_data_active.fillna("").to_csv(f"{syn_sparse_path_active}", index=False)
new_data_inactive.fillna("").to_csv(f"{syn_sparse_path_inactive}", index=False)
print(f"NEW INACTIVE DATA HERE: {new_data_inactive}")

# Combine active/inactive partition
combine_syn_data_loader = CombinePartitionedSyntheticDataLoader(datetime_now, syn_sparse_path_active, syn_sparse_path_inactive)
combined_dense = combine_syn_data_loader.get_dense_combined()
combined_sparse = combine_syn_data_loader.get_sparse_combined()
print(f"Combined data: \n {combined_sparse}")
