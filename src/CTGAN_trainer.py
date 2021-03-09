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
import datetime

# import own scripts
import conf
from check_existence_dir_csv import check_dir, check_csv_path

class TrainModel():
    def __init__(self, dn, ctgan_model_path, nr_epochs):
        # Write training progress to csv file
        '''check_dir(f'{conf.OUTPUT_DIR}/training_logs/')
        self.out_file = f'{conf.OUTPUT_DIR}/training_logs/CTGAN_training_log.csv'
        check_csv_path(self.out_file, ['date', 'dataset_name', 'epoch', 'score'])'''
        
        self.ctgan_model_path = ctgan_model_path
        self.dataset_name = dn

        # Set the hyperparameter values 
        self.epochs = nr_epochs # Default value = 300
        self.batch_size = 200 #500 # Default value = 500 -> needs to be a multiple of 10
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
        current_date = datetime.datetime.now().strftime('%d%m%y_%H%M')
        params = [current_date, time(), self.dataset_name, self.epochs, self.batch_size, self.generator_lr, self.generator_decay, self.discriminator_lr, self.discriminator_decay, self.eval, self.building_time]
        return params

    def build_model(self, data_train, nr_samples=200):
        self.build_timer.start()
        # run block of code and catch warnings
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            
            model = CTGAN(
                epochs=self.epochs
            )
            model.fit(data_train)

            print("CTGAN model was fitted to input data.")
            self.build_timer.stop()
            self.building_time = self.build_timer.get_elapsed_time()
            
            """
            Generate synthetic data from the model.
            """
            self.new_data = model.sample(len(data_train))
            print(f"New generated data: \n {self.new_data}")
            self.eval = self.eval_model(data_train, self.new_data)

            """
            Saved file will not contain any information about the original data.
            Thus, it is safe to share with others.
            """
            model.save(self.ctgan_model_path)
    
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
        


       