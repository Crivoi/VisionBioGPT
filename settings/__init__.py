import logging
import os
from datetime import datetime

from torch import device, cuda

import config
from settings.utils import DataSamples

logger = logging.getLogger(__name__)

DEVICE = device("cuda" if cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

ROOT_DIR = config.ROOT_DIR
BIOGPT_CHECKPOINT = "microsoft/biogpt"
VIT_CHECKPOINT = "google/vit-base-patch16-224"
MAX_SEQ_LENGTH = 1024
NUM_LABELS = 50

DATA_SPLIT = DataSamples.full.value
CACHE_DIR = os.path.join(ROOT_DIR, f'cache_{DATA_SPLIT}_{MAX_SEQ_LENGTH}_seq_len')
DATA_DIR = os.path.join(ROOT_DIR, 'data', DATA_SPLIT)
CXR_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'mimic-cxr')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'model_output', datetime.now().strftime('%m_%d'))

if __name__ == '__main__':
    pass
