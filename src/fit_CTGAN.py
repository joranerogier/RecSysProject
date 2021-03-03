"""
This is the CTGAN Training class module.
The hyperparameters and paths/directories can be changed in the __init__ function if needed
# TODO: Create settings file with the hyperparameter values for a better overview (or give as arguments)
""" 


import csv
import warnings
from sdv.tabular import CTGAN
from sdv.evaluation import evaluate


# import own scripts
import conf
from check_existence_dir_csv import check_dir, check_csv_path

class TrainModel():
    def __init__(self, dn):
        # Write training progress to csv file
        check_dir(f'{conf.OUTPUT_DIR}/training_logs/')
        self.out_file = f'{conf.OUTPUT_DIR}/training_logs/CTGAN_training_log.csv'
        check_csv_path(self.out_file, ['dataset_name', 'batch_size', 'generator_lr', 'generator_decay', 'discriminator_lr', 'discriminator_decay', 'evaluation'])
        
        self.ctgan_dir = f"{conf.OUTPUT_DIR}CTGAN_models/"
        check_dir(self.ctgan_dir)

        self.ctgan_model_path = f'{self.ctgan_dir}test_model.pkl'
        self.dataset_name = dn

        # Set the hyperparameter values 
        self.epochs = 300 # Default value = 300
        self.batch_size = 500 # Default value = 500 -> needs to be a multiple of 10
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
    

    def write_params_csv(self):
        """
        Old function -> reuse if possible to log the training 
        """
        # TODO: Add duration training process + timestamp
        conn = open(self.out_file, 'a')
        writer = csv.writer(conn)
        writer.writerow([self.dataset_name, self.batch_size, self.generator_lr, self.generator_decay, self.discriminator_lr, self.discriminator_decay, self.eval])
        conn.close()

    def build_model(self, data, nr_samples=200):
        # run block of code and catch warnings
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            
            model = CTGAN()
            model.fit(data)
            print("CTGAN model was fitted to input data.")

            """
            Generate synthetic data from the model.
            """
            new_data = model.sample(nr_samples)
            print(f"New generated data: \n {new_data}")

            """
            Saved file will not contain any information about the original data.
            Thus, it is safe to share with others.
            """
            model.save(self.ctgan_model_path)
           

        # Write the used parameters and evaluation to csv file
        #self.write_params_csv()
    
    def eval_model(self, original_data, nr_samples=200):
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            loaded = CTGAN.load(self.ctgan_model_path)
            new_data = loaded.sample(nr_samples)
            eval = evaluate(new_data, original_data)
            return eval
        


       