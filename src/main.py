# TODO: add arguments to change variables directly.
import pandas as pd
import argparse
import datetime

# Import own packages
from CTGAN_trainer import TrainModel
from load_input_data import InputDataLoader
from check_existence_dir_csv import check_dir, check_csv_path
from RecommenderClass import RecommenderSystem
from load_synthetic_data import SyntheticDataLoader
import conf 
from compare_original_synthetic_data import DataComparison

def main(epochs, input_data, model_file_name, input_file, comparison_file_name):
    ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
    check_dir(ctgan_dir)
    ctgan_model_path = f'{ctgan_dir}{model_file_name}'

    comparison_dir = f"{conf.OUTPUT_DIR}comparison_logs/"
    check_dir(comparison_dir)
    comparison_path = f"{comparison_dir}{comparison_file_name}"

    data_loader = InputDataLoader(input_data, input_file)

    input_data = data_loader.get_sparse_data()

    # Build and fit the CTGAN model to the input data
    m = TrainModel(dn="test eval", ctgan_model_path=ctgan_model_path)
    m.build_model(data_train=input_data)

    new_data = m.get_new_data()
    
    df_comparison = DataComparison(input_data, new_data, comparison_path)
    comp = df_comparison.get_comparison_df()
    print(comp)
    df_comparison.write_values_csv()
    
    '''# Load the synthetic data. Nr of samples should be the same as original data.
    nr_samples = len(input_data.columns)
    synthetic_data_loader = SyntheticDataLoader(ctgan_model_path, nr_samples)
    syn_train, syn_test = synthetic_data_loader.get_train_test_data()

    # Apply the recommender system algorithm to the original and new data
    recsys = RecommenderSystem(syn_train, syn_test)
    eval_itemKNN =  recsys.itemKNN()
    print(f"Evaluation itemKNN: {eval_itemKNN}")    
    eval_userKNN = recsys.userKNN()
    print(f"Evaluation userKNN: {eval_userKNN}")'''

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--epochs", type=int, help="Nr. of epochs for training the CTGAN model", default=1)
    ap.add_argument("-D", "--data", type=str, help="Which input data to use: own, ml-100k, demo", default="own")
    ap.add_argument("-M", "--ctgan_model_name", type=str, help="Path to saving location of trained ctgan", default="test_model.pkl")
    ap.add_argument("-I", "--input_file_name", type=str, help="Filename of own input data", default="mini_ml100k_user_10_item_25.csv")
    ap.add_argument("-C", "--comparison_file_name", type=str, help="Filename for csv output comparison data", default="test.csv")
    args = vars(ap.parse_args())

    nr_epochs = args['epochs']
    input_data = args['data']
    model_name = args['ctgan_model_name']
    input_file_name = f"{conf.DATA_DIR}/{args['input_file_name']}"

    comparison_file_name = args['comparison_file_name']

    main(nr_epochs, input_data, model_name, input_file_name, comparison_file_name)
