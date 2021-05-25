import os
from check_existence_dir_csv import check_dir

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

OUTPUT_DIR = f'{ROOT_DIR}/output/'
check_dir(OUTPUT_DIR)

DATA_DIR = f'{ROOT_DIR}/data/'

SYN_DATA_DIR = f'{OUTPUT_DIR}synthetic_data/'
check_dir(SYN_DATA_DIR)

PARTITIONED_DATA_DIR = f'{OUTPUT_DIR}partitioned_data/'
PARTITIONED_MAINSTREAM_DATA_DIR = f'{OUTPUT_DIR}partitioned_mainstreaminess_data/'

check_dir(PARTITIONED_DATA_DIR)