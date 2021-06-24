from sdv.tabular import CTGAN
import numpy as np
import torch
import pandas as pd 
import random

# own scripts
import conf
from load_synthetic_data_recombine_partition import CombinePartitionedSyntheticDataLoader
from transform_data_representation import transform_sparse_to_dense_data

prefixes = ['all', 'beyond', 'combined_middle_tail', 'all_middle_tail', 'beyond_middle_tail']
new_sampling_range = "m100_l300"
epochs = 750

model_file_name = f"_tau_0.165_{epochs}eps_300bs_str3"
datetime_now = '140621_1007'
ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
all_ctgan_model_path = f'{ctgan_dir}{prefixes[0]}{model_file_name}.pkl'
beyond_mainstream_ctgan_model_path = f'{ctgan_dir}{prefixes[1]}{model_file_name}.pkl'

# path to where to save new synthetic data
syn_sparse_path_all = f'{conf.SYN_DATA_DIR}syn_sparse_{prefixes[3]}{model_file_name}.csv'
syn_sparse_path_beyond_mainstream = f'{conf.SYN_DATA_DIR}syn_sparse_{prefixes[4]}{model_file_name}.csv'
syn_sparse_path_combined = f'{conf.SYN_DATA_DIR}syn_sparse_{prefixes[2]}{model_file_name}_{new_sampling_range}.csv'
syn_dense_path_combined = f'{conf.SYN_DATA_DIR}syn_dense_{prefixes[2]}{model_file_name}_{new_sampling_range}.csv'

# Original training data (partitioned)
orig_sparse_all = f"{conf.PARTITIONED_MAINSTREAM_DATA_DIR}orig_sparse_all_tau_0.165_usercolumnremoved.csv" #orig_sparse_all_tau_tau_0.07_l20.csv"
orig_sparse_beyond_mainstream = f"{conf.PARTITIONED_MAINSTREAM_DATA_DIR}orig_sparse_beyond_mainstream_tau_0.165_usercolumnremoved.csv"
df_orig_all = pd.read_csv(orig_sparse_all, sep=',', encoding="latin-1").fillna(0)
df_orig_beyond = pd.read_csv(orig_sparse_beyond_mainstream, sep=',', encoding="latin-1").fillna(0)

# Set seed to ensure reproducibility
torch.manual_seed(0)
np.random.seed(1)

# Load the saved models
all_model = CTGAN.load(all_ctgan_model_path)
beyond_mainstream_model = CTGAN.load(beyond_mainstream_ctgan_model_path)

# Create new data & save
new_data_all = all_model.sample(len(df_orig_all))
new_data_beyond = beyond_mainstream_model.sample(len(df_orig_beyond))
#print(f"NEW ALL DATA HERE: \n{new_data_all}")

# Drop len(orig_sparse_beyond_mainstream) user from the "all" data FROM THE MIDDLE TAIL to end with the same amount of users as the original data
print(f"Length beyond partition: {len(new_data_beyond)}")

# Group original and synthetic data by users
new_data_all_dense = transform_sparse_to_dense_data(new_data_all)
new_data_all_dense = new_data_all_dense[new_data_all_dense.rating != 0]
#new_data_all_dense = pd.read_csv('dense_all_165.csv', encoding="latin-1")
print(new_data_all_dense)
new_data_all_dense.to_csv('dense_all_165.csv', index=False)
df_all_users = new_data_all_dense.groupby('user').size().reset_index(name='counts')

print(df_all_users)
df_all_users_sub = df_all_users[(df_all_users['counts'] > 100) & (df_all_users['counts'] < 200)]
print(df_all_users_sub)

# take random indexes from subset
random_indexes = random.sample(range(len(df_all_users_sub)), len(df_orig_beyond))
print(random_indexes)
#user_ids_to_drop = [x+1 for x in random_indexes] # user_ids start at 1, indexes at 0 -> DO NOT DO THIS -> removed by indexes in next part

# drop rows based on user ids
print(new_data_all)
new_data_all.drop(random_indexes, 0, inplace=True)

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