import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

conn = psycopg2.connect(
    database='mimic',
    user=POSTGRES_USERNAME,
    password=POSTGRES_PASSWORD,
    host='localhost',
    port='5432'
)

cursor = conn.cursor()

if __name__ == '__main__':
    pass
