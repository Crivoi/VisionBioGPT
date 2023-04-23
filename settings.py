import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

db_kwargs = dict(
    database="mydb",
    user=POSTGRES_USERNAME,
    password=POSTGRES_PASSWORD,
    host="localhost",
    port="5432"
)

conn = psycopg2.connect(**db_kwargs)

cur = conn.cursor()

if __name__ == '__main__':
    pass
