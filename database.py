from functools import cached_property
from typing import Union

import psycopg2
from psycopg2 import sql

from settings import POSTGRES_DB_NAME, POSTGRES_USERNAME, POSTGRES_PASSWORD


class MimicDatabase:
    """
    Representation of MIMIC-III local Postgres DB

    Notes:
        - identifies patients based on subject_id and hadm_id as a tuple
        - since we are only interested in using discharge summaries, it is important to note that there are no erroneous
          cases (where mimiciii.noteevents.iserror = '1'), thus it is not needed to filter for errors
    """

    def __init__(self):
        self.category = 'Discharge summary'

        self._conn = psycopg2.connect(
            database=POSTGRES_DB_NAME,
            user=POSTGRES_USERNAME,
            password=POSTGRES_PASSWORD,
            host='localhost',
            port='5432'
        )
        self._cursor = self._conn.cursor()

    @cached_property
    def top_icd9_codes(self):
        self.execute_query("""
            SELECT icd9_code FROM mimiciii.diagnoses_icd 
            GROUP BY icd9_code ORDER BY count(*) DESC LIMIT 50
        """)
        return [r[0] for r in self.fetchall()]

    @cached_property
    def count_discharge_summaries(self):
        """
        :return: Total number of discharge summaries linked to an icd9_code via subject_id and hadm_id
        """
        query = """
            SELECT count(*)
            FROM mimiciii.diagnoses_icd d 
            JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
            AND d.hadm_id = n.hadm_id
            AND n.category = %s
            AND d.icd9_code IN %s
        """
        self.execute_query(query, [self.category, tuple(self.top_icd9_codes)])
        return self.fetchone()[0]

    @cached_property
    def count_hadm_ids(self):
        """
        :return: Number of distinct hospital admission ids
        """
        query = """
            SELECT count(distinct d.hadm_id)
            FROM mimiciii.diagnoses_icd d 
            JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
            AND d.hadm_id = n.hadm_id
            AND n.category = %s
            AND d.icd9_code IN %s
        """
        self.execute_query(query, [self.category, tuple(self.top_icd9_codes)])
        return self.fetchone()[0]

    @cached_property
    def count_subject_ids(self):
        """
        :return: Number of distinct hospital admission ids
        """
        query = """
            SELECT count(distinct d.hadm_id)
            FROM mimiciii.diagnoses_icd d 
            JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
            AND d.hadm_id = n.hadm_id
            AND n.category = %s
            AND d.icd9_code IN %s
        """
        self.execute_query(query, [self.category, tuple(self.top_icd9_codes)])
        return self.fetchone()[0]

    @cached_property
    def count_icd9_codes(self):
        query = """
            SELECT count(distinct d.icd9_code)
            FROM mimiciii.diagnoses_icd d 
            JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
            AND d.hadm_id = n.hadm_id
            AND n.category = %s
            AND d.icd9_code IN %s
        """
        self.execute_query(query, [self.category, tuple(self.top_icd9_codes)])
        return self.fetchone()[0]

    def query_text_and_icd9_code(self):
        query = """
            SELECT n.text, d.icd9_code
            FROM mimiciii.diagnoses_icd d 
            JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
            AND d.hadm_id = n.hadm_id
            AND n.category = %s
            AND d.icd9_code IN %s
        """
        self.execute_query(query, [self.category, tuple(self.top_icd9_codes)])

    def execute_query(self, query: Union[str, sql.SQL], query_args: list = None):
        return self._cursor.execute(query, query_args)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchmany(self, size: int = 32):
        return self._cursor.fetchmany(size)

    def fetchall(self):
        return self._cursor.fetchall()


if __name__ == '__main__':
    database: MimicDatabase = MimicDatabase()
    print('Top Codes:', database.top_icd9_codes)
    print('Total discharge summaries:', database.count_discharge_summaries)
    print('# Distinct Subject ids:', database.count_subject_ids)
    print('# Distinct Hadm ids:', database.count_hadm_ids)
    print('# Distinct labels:', database.count_icd9_codes)
