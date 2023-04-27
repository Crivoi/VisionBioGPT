import os

from torch import device, cuda
from dotenv import load_dotenv

load_dotenv()

MIMIC_PATH = os.getenv('MIMICIII_PATH', None)

POSTGRES_DB_NAME = os.getenv('POSTGRES_DB_NAME', 'mimic')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

DEVICE = device("cuda" if cuda.is_available() else "cpu")
