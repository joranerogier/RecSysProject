# TODO: add arguments to change variables directly.
import pandas as pd
import argparse

# Import own packages
from CTGAN_trainer import TrainModel
from load_input_data import InputDataLoader
from check_existence_dir_csv import check_dir, check_csv_path
from RecommenderClass import RecommenderSystem
import conf 

def main(epochs, input_data, model_file_name):
    ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
    check_dir(ctgan_dir)
    ctgan_model_path = f'{ctgan_dir}{model_file_name}'

    data_loader = InputDataLoader("ml-100k")
    train_sparse = data_loader.get_train_sparse_data()
    test_sparse = data_loader.get_test_sparse_data()
    train_rec = data_loader.get_train_rec_data()
    test_rec = data_loader.get_test_rec_data()

    # Build and fit the CTGAN model to the input data
    m = TrainModel(dn="test eval", ctgan_model_path=ctgan_model_path)
    m.build_model(data_train=train_sparse)

    # Apply the recommender system algorithm to the original and new data
    recsys = RecommenderSystem(train_rec, test_rec)
    eval_itemKNN =  recsys.itemKNN()
    print(f"Evaluation itemKNN: {eval_itemKNN}")    
    eval_userKNN = recsys.userKNN()
    print(f"Evaluation userKNN: {eval_userKNN}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--epochs", type=int, help="Nr. of epochs for training the CTGAN model", default=1)
    ap.add_argument("-D", "--data", type=str, help="Which input data to use: own, ml-100k, demo", default="own")
    ap.add_argument("-M", "--ctgan_model_name", type=str, help="Path to saving location of trained ctgan", default="test_model.pkl")
    args = vars(ap.parse_args())
    nr_epochs = args['epochs']
    input_data = args['data']
    model_name = args['ctgan_model_name']
    main(nr_epochs, input_data, model_name)
