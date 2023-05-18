import os

from torch import device, cuda
from dotenv import load_dotenv

load_dotenv()

MIMIC_PATH = os.getenv('MIMICIII_PATH', None)

POSTGRES_DB_NAME = os.getenv('POSTGRES_DB_NAME', 'mimic')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

DEVICE = device("cuda" if cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

MODEL_SAVE_DIR = 'model-test'

BIOGPT_CHECKPOINT = "microsoft/biogpt"
MAX_SEQ_LENGTH = 5
NUM_LABELS = 50
BATCH_SIZE = 8
