import os
import logging

from torch import device, cuda
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MIMIC_PATH = os.getenv('MIMICIII_PATH', None)

POSTGRES_DB_NAME = os.getenv('POSTGRES_DB_NAME', 'mimic')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

DEVICE = device("cuda" if cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

MODEL_SAVE_DIR = 'model-test'

BIOGPT_CHECKPOINT = "microsoft/biogpt"
MAX_SEQ_LENGTH = 1024
NUM_LABELS = 50
BATCH_SIZE = 8
