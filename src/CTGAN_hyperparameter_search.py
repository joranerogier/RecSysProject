"""

""" 

import csv
import warnings
from sdv.tabular import CTGAN
from sdv.metrics.tabular import SVCDetection, CSTest
from sdv.evaluation import evaluate
from timer import Timer
from time import time
import numpy as np
import torch
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import make_scorer
import skopt

# import own scripts
import conf
from check_existence_dir_csv import check_dir, check_csv_path
from load_input_data import InputDataLoader

input_data = 'ml-100k'

#TODO: make this variable optional
input_file_name = f"{conf.DATA_DIR}/None" # not used for hyperparameter tuningn on ml-100k dataset

data_loader = InputDataLoader(input_data, input_file_name)
input_data = data_loader.get_sparse_data()

# Set the hyperparameter values 
epochs = 1 # Default value = 300        
SPACE = [skopt.space.Categorical(['200', '300', '400', '500', '600', '700'], name='batch_size'),
        skopt.space.Real(0.00001, 0.5, name='generator_lr', prior='log-uniform'),
        skopt.space.Real(0.0000001, 0.01, name='generator_decay', prior='log-uniform'),
        skopt.space.Real(0.00001, 0.5, name='discriminator_lr', prior='log-uniform'),
        skopt.space.Real(0.0000001, 0.01, name='discriminator_decay', prior='log-uniform')]

#All default values (for now)
#self.log_frequency = True #  check what this does, and if need to adjust or not
#self.embedding_dim = 128  # Size of the random sample passed to the Generator
#self.generator_dim = (256,256) 
#self.discriminator_dim = (256,256)
#self.discriminator_steps = 1 # 1 to match CTGAN implementation (5 for matching WGAN)

eval = 0
build_timer = Timer()
building_time = None
new_data = None

def compute_score(orig, syn):
    """
    Function to compute score used for comparing various models when searching the optimal hyperparameters.
    Metrics used: SVC Detection & CSTest
    """
    #svc_score = SVCDetection.compute(orig, syn) # returns 1 - AUROC
    chtest_score = CSTest.compute(orig, syn)# goal: maximize

    comb_score = evaluate(syn, orig, metrics=['CSTest', 'SVCDetection'])
    print(orig.head())
    print(syn.head())
    return chtest_score


@skopt.utils.use_named_args(SPACE)
def evaluate_model(**params):
    all_params = {**params}
    print(all_params)

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        # Set seed to ensure reproducibility
        torch.manual_seed(0)
        np.random.seed(0)

        model = CTGAN(
            epochs=epochs,
            batch_size=int(all_params['batch_size']),
            generator_lr=all_params['generator_lr'],
            generator_decay=all_params['generator_decay'],
            discriminator_lr=all_params['discriminator_lr'],
            discriminator_decay=all_params['discriminator_decay']
        )
        print(f"Batch size: {all_params['batch_size']},\n Generator lr: {all_params['generator_lr']},\n Generator decay: {all_params['generator_decay']},\n Discriminator lr: {all_params['discriminator_lr']},\n Discriminator decay: {all_params['discriminator_decay']}")
        model.fit(input_data)
        new_data = model.sample(len(input_data))
        scorer = make_scorer(compute_score, greater_is_better=True)
        
        result = -np.mean(cross_val_score(model, input_data, new_data,  cv=3, n_jobs=-1, scoring=scorer))
        print(f"Result cross validation: {result}")
    return result

# perform optimization
result = skopt.gp_minimize(evaluate_model, SPACE)        
print(f"Best score = {result.fun}")
print(f"Best parameters: \n - batch size: {result.x[0]},\n - gen. lr: {result.x[1]},\n gen. decay: {result.x[2]},\n discr. lr: {result.x[3]},\n discr. decay: {result.x[4]}")

       