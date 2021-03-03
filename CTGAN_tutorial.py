from sdv.demo import load_tabular_demo
from sdv.tabular import CTGAN
from os import path
import os
import warnings
import conf
from check_existence_dir import check_dir

# run block of code and catch warnings
with warnings.catch_warnings():
    # ignore all caught warnings
    warnings.filterwarnings("ignore")
    
    """
    Check if output (& CTGAN models) directory exists. If not, create one.
    """
    wd = os.getcwd()
    output_dir = f"{wd}/output/"
    ctgan_dir = f"{output_dir}CTGAN_models/"
    check_dir(output_dir)
    check_dir(ctgan_dir)

    ctgan_test_model = f'{ctgan_dir}test_model.pkl'

    data = load_tabular_demo('student_placements_pii')
    # data.head()

    model = CTGAN(
        # primary_key indicates the name of the column that is the index of the table
        # ensuring that no double values will be generated for this variable.
        primary_key = 'student_id',

        # anonymize_fields to mask Personal Identifiable Information (PII) in the dataset
        anonymize_fields={
            'address': 'address'
        }
    )
    """ 
    The model fitting process takes care of transforming the different fields using the appropriate
    'Reversible Data Transforms to ensure tha tthe data has a format that underlying 
    CTGANSynthesizer class can handle.'
    """
    model.fit(data)
    print("CTGAN model was fitted demo data.")

    """
    Generate synthetic data from the model.
    """
    new_data = model.sample(200)
    print(f"New generated data: \n {new_data}")

    """
    Saved file will not contain any information about the original data.
    Thus, it is safe to share with others.
    """
    model.save(ctgan_test_model)

    """
    Load the model and synthetic data.
    """
    loaded = CTGAN.load(ctgan_test_model)
    