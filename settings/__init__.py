import os
import logging
from datetime import datetime

from torch import device, cuda

import config
# from dotenv import load_dotenv

from settings.utils import DataSamples

# load_dotenv()
logger = logging.getLogger(__name__)

# MIMIC_PATH = os.getenv('MIMICIII_PATH', None)
# POSTGRES_DB_NAME = os.getenv('POSTGRES_DB_NAME', 'mimic')
# POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME', 'postgres')
# POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

DEVICE = device("cuda" if cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

ROOT_DIR = config.ROOT_DIR
BIOGPT_CHECKPOINT = "microsoft/biogpt"
MAX_SEQ_LENGTH = 1024
NUM_LABELS = 50

DATA_SPLIT = DataSamples.sample.value
CACHE_DIR = os.path.join(ROOT_DIR, f'cache_{DATA_SPLIT}_{MAX_SEQ_LENGTH}_seq_len')
DATA_DIR = os.path.join(ROOT_DIR, 'data', DATA_SPLIT)
OUTPUT_DIR = os.path.join(ROOT_DIR, 'model_output', datetime.now().strftime('%m_%d'))

if __name__ == '__main__':
    print(ROOT_DIR)
