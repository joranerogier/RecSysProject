import os
from check_existence_dir_csv import check_dir

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

OUTPUT_DIR = f'{ROOT_DIR}/output/'
check_dir(OUTPUT_DIR)

DATA_DIR = f'{ROOT_DIR}/data/'