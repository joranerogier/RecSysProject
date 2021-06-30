"""
Script used to create new samples based on a provided CTGAN model.
To create as the same amount of samples as the original data the model was trained on, 
the original dataset should be provided as well.
"""

from sdv.tabular import CTGAN
import numpy as np
import torch
import pandas as pd 
from lenskit.datasets import ML100K
import argparse

# own scripts
import conf
from load_input_data import InputDataLoader

def main(s):
    part = 'all_' # all_ or beyond_ or ''

    sample_subset_ratio = 1
    model_file_name = f"{part}tau_0.18_impl60_750eps_300bs"
    ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
    ctgan_model_path = f'{ctgan_dir}{model_file_name}.pkl'


    # path to where to save new synthetic data
    syn_sparse_path = f'{conf.SYN_DATA_DIR}{part}tau_0.18_impl60_750eps_300bs.csv'

    # Original training data (sparse)
    data_loader = InputDataLoader(s, "own", f"{conf.OUTPUT_DIR}partitioned_mainstreaminess_data/orig_sparse_{part}tau_0.18_playcounts_impl60_lastfm_items_removed_implicit_true.csv")
    input_data = data_loader.get_sparse_data()

    # Set seed to ensure reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Load the saved model
    m = CTGAN.load(ctgan_model_path)

    # Create new data & save
    sampling_set = int(len(input_data)*sample_subset_ratio)
    new_data = m.sample(sampling_set)
    new_data.fillna(0).to_csv(f"{syn_sparse_path}", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-S", "--separation", type=str, help="separation symbol of csv file: ; or ,", default=',')
    args = vars(ap.parse_args())

    main(args['separation'])
