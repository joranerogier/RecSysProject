# TODO: add arguments to change variables directly.
import pandas as pd
import argparse
import csv
import datetime

# Import own packages
from CTGAN_trainer import TrainModel
from load_input_data import InputDataLoader
from  load_input_data_partition_active_vs_inactive import PartitionedInputDataLoader
from check_existence_dir_csv import check_dir, check_csv_path
from RecommenderClass import RecommenderSystem
from load_synthetic_data import SyntheticDataLoader
import conf 
from compare_original_synthetic_data import DataComparison
from load_synthetic_data_recombine_partition import CombinePartitionedSyntheticDataLoader

def write_to_csv(file_path, values):
    with open(file_path, mode="a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(values)
    f.close()


def main(epochs, input_data, model_file_name, input_file, comparison_file_name, p, bs, lc):
    # Check and set directories/paths for the CTGAN models
    
    # Set directory to save CTGAN to /scratch directory, due to memory errors on own home directory
    if lc == 'cluster':
        ctgan_dir = f"/scratch/jrogier/"
    else: 
        ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
    check_dir(ctgan_dir)

    # Check and set directories/paths for the training & comparison results
    check_dir(f'{conf.OUTPUT_DIR}/training_comparison_logs/')
    logging_file = f'{conf.OUTPUT_DIR}/training_comparison_logs/{comparison_file_name}'
    check_csv_path(logging_file, ['date', 'epoch_date', 'dataset_name', 'nr_epochs', 'batch_size', 'generator_lr', 'generator_decay', 'discriminator_lr', 'discriminator_decay', 'evaluation', 'fitting time (s)', 'nr_users_orig', 'nr_user_syn', 'nr_items_orig', 'nr_items_syn', 'sparseness_orig', 'sparseness_syn'])
    datetime_now = datetime.datetime.now().strftime('%d%m%y_%H%M')
    
    # Set path for sparse active & inactive partition csv
    orig_sparse_path_active = f'{conf.PARTITIONED_DATA_DIR}orig_sparse_{datetime_now}_active.csv'
    orig_sparse_path_inactive = f'{conf.PARTITIONED_DATA_DIR}orig_sparse_{datetime_now}_inactive.csv'
    syn_sparse_path_active = f'{conf.SYN_DATA_DIR}syn_sparse_{datetime_now}_active.csv'
    syn_sparse_path_inactive = f'{conf.SYN_DATA_DIR}syn_{datetime_now}_inactive.csv'

    if (p==False):
        # For non-partitioned data
        data_loader = InputDataLoader(input_data, input_file)
        input_data = data_loader.get_sparse_data()

        # Build and fit the CTGAN model to the input data
        ctgan_model_path = f'{ctgan_dir}{model_file_name}.pkl'
        m = TrainModel(dn="test", ctgan_model_path=ctgan_model_path, nr_epochs=epochs, curr_date=datetime_now, batch_size=bs)
        m.build_model(data_train=input_data)

        new_data = m.get_new_data()

        # Get and print comparison dataframe
        df_comparison = DataComparison(input_data, new_data)
        comp = df_comparison.get_comparison_df()
        print(comp)

    else:
        # complete input data
        data_loader_c = InputDataLoader(input_data, input_file)
        input_data_c = data_loader_c.get_sparse_data()

        # partitioned input data & Save to csv file
        data_loader = PartitionedInputDataLoader(input_data, input_file)
        input_data_active, input_data_inactive = data_loader.get_sparse_data()
        input_data_active.fillna("").to_csv(f"{orig_sparse_path_active}", index=False)
        input_data_inactive.fillna("").to_csv(f"{orig_sparse_path_inactive}", index=False)

        active_ctgan_model_path = f'{ctgan_dir}{model_file_name}_active.pkl'
        inactive_ctgan_model_path = f'{ctgan_dir}{model_file_name}_inactive.pkl'
        
        # CTGAN model for active users
        m_active = TrainModel(dn="test eval", ctgan_model_path=active_ctgan_model_path, nr_epochs=epochs, curr_date=datetime_now, batch_size=bs)
        m_active.build_model(data_train=input_data_active, user_part="active")
        new_data_active = m_active.get_new_data()
        print(f"NEW DATA HERE: {new_data_active}")
        new_data_active.fillna("").to_csv(f"{syn_sparse_path_active}", index=False)
        
        # CTGAN model for inactive users
        m_inactive = TrainModel(dn="test eval", ctgan_model_path=inactive_ctgan_model_path, nr_epochs=epochs, curr_date=datetime_now, batch_size=bs)
        m_inactive.build_model(data_train=input_data_inactive, user_part="inactive")
        new_data_inactive = m_inactive.get_new_data()
        new_data_inactive.fillna("").to_csv(f"{syn_sparse_path_inactive}", index=False)

        # Recombine active and inactive users datasets
        # check if there are no duplicate users in the two subsets
        # TODO: How to deal with items between the two subsets? There is no guarantee that they are the same.
        # -> if they are not in one of the two subsets originally, it is no problem that they are not in the combined dataframe for those users either.
        combine_syn_data_loader = CombinePartitionedSyntheticDataLoader(datetime_now, syn_sparse_path_active, syn_sparse_path_inactive)
        combined_dense = combine_syn_data_loader.get_dense_combined()
        combined_sparse = combine_syn_data_loader.get_sparse_combined()

        # Get and print comparison dataframe
        df_comparison = DataComparison(input_data_c, combined_sparse)
        comp = df_comparison.get_comparison_df()
        print(comp)
        

    return

    
    '''# Log training and comparison results
    training_values = m.get_params_csv()
    comparison_values = df_comparison.get_values_csv()
    all_values = training_values + comparison_values
    write_to_csv(logging_file, all_values)'''
    
    # Load the synthetic data (split in train & test set)
    #nr_samples = len(input_data)
    #synthetic_data_loader = SyntheticDataLoader(datetime_now, epochs, bs, "")
    #syn_train, syn_test = synthetic_data_loader.get_train_test_data()

    # Apply the recommender system algorithm to the original and new data
    #recsys = RecommenderSystem(syn_train, syn_test)
    #eval_itemKNN =  recsys.itemKNN()
    #print(f"Evaluation itemKNN: {eval_itemKNN}")    
    #eval_userKNN = recsys.userKNN()
    #print(f"Evaluation userKNN: {eval_userKNN}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--epochs", type=int, help="Nr. of epochs for training the CTGAN model", default=1)
    ap.add_argument("-D", "--data", type=str, help="Which input data to use: own, ml-100k, demo", default="own")
    ap.add_argument("-M", "--ctgan_model_name", type=str, help="Path filename to saving location of trained ctgan", default="test_model")
    ap.add_argument("-I", "--input_file_name", type=str, help="Filename of own input data", default="mini_ml100k_user_10_item_25.csv")
    ap.add_argument("-C", "--comparison_file_name", type=str, help="Filename for csv output comparison data", default="test.csv")
    ap.add_argument("-P", "--partition_active_inactive", type=bool, help="Boolean, telling of the data should be partitioned in active/inactive users", default=False )
    ap.add_argument("-BS", "--batch_size", type=int, help="batch size during training CTGAN", default=500)
    ap.add_argument("-LC", "--local_or_cluster", type=str, help="(remote)machine where the scripts are run", default='local')   
    args = vars(ap.parse_args())

    nr_epochs = args['epochs']
    batch_size = args['batch_size']
    input_data = args['data']
    model_name = args['ctgan_model_name']
    input_file_name = f"{conf.DATA_DIR}/{args['input_file_name']}"

    comparison_file_name = args['comparison_file_name']
    local_or_cluster = args['local_or_cluster']
    partition = args['partition_active_inactive']

    main(nr_epochs, input_data, model_name, input_file_name, comparison_file_name, partition, batch_size, local_or_cluster)
