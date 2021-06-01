"""
This is the CTGAN Training class module.
The hyperparameters and paths/directories can be changed in the __init__ function if needed
# TODO: Create settings file with the hyperparameter values for a better overview (or give as arguments)
""" 
import csv
import warnings
from sdv.tabular import CTGAN
from sdv.evaluation import evaluate
from timer import Timer
from time import time
import numpy as np
import torch

# import own scripts
import conf
from check_existence_dir_csv import check_dir, check_csv_path

class TrainModel():
    def __init__(self, dn, ctgan_model_path, nr_epochs, curr_date, batch_size):
        self.current_date = curr_date
        self.ctgan_model_path = ctgan_model_path
        self.dataset_name = dn

        # Set the hyperparameter values 
        self.epochs = nr_epochs # Default value = 300
        self.batch_size = batch_size #500 # Default value = 500 -> needs to be a multiple of 10
        self.generator_lr = 2e-4 # Default value = 2e-4
        self.generator_decay = 1e-6 # Default value = 1e-6
        self.discriminator_lr = 2e-4 # Default value = 2e-4
        self.discriminator_decay = 1e-6 # Default value = 1e-6
        
        #All default values (for now)
        #self.log_frequency = True #  check what this does, and if need to adjust or not
        #self.embedding_dim = 128  # Size of the random sample passed to the Generator
        #self.generator_dim = (256,256) 
        #self.discriminator_dim = (256,256)
        #self.discriminator_steps = 1 # 1 to match CTGAN implementation (5 for matching WGAN)
        
        self.eval = 0
        self.build_timer = Timer()
        self.building_time = None
        self.new_data = None

    def get_params_csv(self):
        params = [self.current_date, time(), self.dataset_name, self.epochs, self.batch_size, self.generator_lr, self.generator_decay, self.discriminator_lr, self.discriminator_decay, self.eval, self.building_time]
        return params

    def build_model(self, data_train, nr_samples=200, user_part=""):
        self.build_timer.start()
        # run block of code and catch warnings
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")

            # Set seed to ensure reproducibility
            torch.manual_seed(0)
            np.random.seed(0)

            model = CTGAN(
                epochs=self.epochs, 
                batch_size=self.batch_size,
                verbose=True,
                log_frequency= True,
                field_transformers = {'one_hot_encoding': "OneHotEncodingTransformer"}
            )
            model.fit(data_train)

            print("CTGAN model was fitted to input data.")
            self.build_timer.stop()
            self.building_time = self.build_timer.get_elapsed_time()
            
            """
            Generate synthetic data from the model.
            """
            self.new_data = model.sample(len(data_train))
            print(f"New generated data [partition = {user_part}]: \n {self.new_data}")

            # check if active/inactive user partition is used, to save the file accordingly
            if (user_part == ""):
                self.new_data.fillna(0).to_csv(f"{conf.SYN_DATA_DIR}syn_sparse_{self.current_date}.csv", index=False)
            elif (user_part == "active"):
                self.new_data.fillna(0).to_csv(f"{conf.SYN_DATA_DIR}syn_sparse_{self.current_date}_active.csv", index=False)
            elif (user_part == "inactive"):
                self.new_data.fillna(0).to_csv(f"{conf.SYN_DATA_DIR}syn_sparse_{self.current_date}_inactive.csv", index=False)
            else:
                print("[WARNING] Synthetic data is not saved to file.")
            #self.eval = self.eval_model(data_train, self.new_data)

            """
            Saved file will not contain any information about the original data.
            Thus, it is safe to share with others.
            """
            model.save(self.ctgan_model_path)
            print(f"Model is saved to: {self.ctgan_model_path}")
    
    def eval_model(self, original_data, new_data):
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            #loaded = CTGAN.load(self.ctgan_model_path)
            #new_data = loaded.sample(nr_samples)
            eval = evaluate(new_data, original_data)
            print(f"evaluation of the model:\n {eval}")
            return eval

    def get_new_data(self):
        return self.new_data
        


       